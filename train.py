import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import argparse
import os
from omegaconf import OmegaConf
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(str(Path(__file__).parent / "stofnet"))
sys.path.append(str(Path(__file__).parent.parent))

from models import StofNet, ZonziniNetLarge, SincNet
from dataloaders.dataset_pala_rf import InSilicoDatasetRf
from utils.mask2samples import samples2mask, samples2nested_list, samples2coords
from utils.gaussian import gaussian_kernel
from utils.hilbert import hilbert_transform
from utils.metrics import toa_rmse
from utils.threshold import find_threshold
from utils.plotting import wb_img_upload, plot_channel_overview
from utils.transforms import NormalizeVol, RandomVol
from utils.collate_fn import collate_fn


# load config
script_path = Path(__file__).parent.resolve()
cfg = OmegaConf.load(str(script_path / 'config.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

unravel_batch_dim = lambda x: x.reshape(cfg.batch_size, x.shape[0]//cfg.batch_size, -1)

# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

if cfg.device == "cuda":
    pin_memory = True
else:
    pin_memory = False

# load dataset
dataset = InSilicoDatasetRf(
    dataset_path = cfg.data_dir,
    sequences = cfg.sequences,
    rescale_factor = cfg.rf_scale_factor,
    ch_gap = cfg.ch_gap,
    angle_threshold = cfg.angle_threshold,
    clutter_db = cfg.clutter_db,
    temporal_filter_opt=cfg.temporal_filter,
    pow_law_opt = cfg.pow_law_opt,
    transforms = torch.nn.Sequential(NormalizeVol()),
    )

# wave compounding indices
angles_list = dataset.get_key('angles_list')
wv_idcs = range(len(angles_list))
wv_idx = 1

# data-related config
cfg.fs = float(dataset.get_key('fs'))
cfg.c = float(dataset.get_key('c'))
cfg.wavelength = float(dataset.get_key('wavelength'))
channel_num = dataset.get_channel_num()
sample_num = dataset.get_sample_num()

# split into train / validation partitions
val_percent = 0.2
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

# create data loaders
num_workers = min(4, os.cpu_count())
loader_args = dict(batch_size=cfg.batch_size, num_workers=num_workers, pin_memory=pin_memory)
train_loader = DataLoader(train_set, collate_fn=collate_fn, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, collate_fn=collate_fn, shuffle=False, drop_last=True, **loader_args)

# instantiate logging
if cfg.logging:
    wb = wandb.init(project='StofNet', resume='allow', anonymous='must', config=cfg)
    wb.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr, val_percent=val_percent))
    wandb.define_metric('train_loss', step_metric='train_step')
    wandb.define_metric('train_points', step_metric='train_step')
    wandb.define_metric('val_loss', step_metric='val_step')
    wandb.define_metric('val_points', step_metric='val_step')
    wandb.define_metric('val_toa_distance', step_metric='val_step')
    wandb.define_metric('val_toa_precision', step_metric='val_step')
    wandb.define_metric('val_toa_recall', step_metric='val_step')
    wandb.define_metric('val_toa_jaccard', step_metric='val_step')
    wandb.define_metric('val_toa_true_positive', step_metric='val_step')
    wandb.define_metric('val_toa_false_positive', step_metric='val_step')
    wandb.define_metric('val_toa_false_negative', step_metric='val_step')
    wandb.define_metric('val_toa_false_negative', step_metric='val_step')
    wandb.define_metric('val_ideal_threshold', step_metric='val_step')
    wandb.define_metric('lr', step_metric='epoch')

# load model
if cfg.model.lower() == 'stofnet':
    model = StofNet(upsample_factor=cfg.upsample_factor, hilbert_opt=cfg.hilbert_opt, concat_oscil=cfg.oscil_opt)
elif cfg.model.lower() == 'zonzini':
    model = ZonziniNetLarge()
elif cfg.model.lower() == 'sincnet':
    cfg.upsample_factor = 1
    sincnet_params = {'input_dim': sample_num*cfg.rf_scale_factor,
                        'fs': cfg.fs,
                        'cnn_N_filt': [128, 128, 128, 1],
                        'cnn_len_filt': [1023, 11, 9, 7],
                        'cnn_max_pool_len': [1, 1, 1, 1],
                        'cnn_use_laynorm_inp': False,
                        'cnn_use_batchnorm_inp': False,
                        'cnn_use_laynorm': [False, False, False, False],
                        'cnn_use_batchnorm': [True, True, True, True],
                        'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
                        'cnn_drop': [0.0, 0.0, 0.0, 0.0],
                        'use_sinc': True,
                        }
    model = SincNet(sincnet_params)
else:
    raise Exception('Model not recognized')

model = model.to(cfg.device)
model.eval()

if cfg.model_file:
    state_dict = torch.load(str(script_path / 'ckpts' / cfg.model_file), map_location=cfg.device)
    model.load_state_dict(state_dict)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

# loss settings
loss_mse = nn.MSELoss(reduction='mean')
loss_l1 = nn.L1Loss(reduction='mean')
zero_l1 = torch.zeros((cfg.batch_size*channel_num, sample_num*cfg.rf_scale_factor*cfg.upsample_factor), device=cfg.device, dtype=torch.float32)
loss_l1_arg = lambda y: loss_l1(y, zero_l1)
gauss_kernel_1d = torch.tensor(gaussian_kernel(size=cfg.kernel_size, sigma=cfg.sigma), dtype=torch.float32, device=cfg.device).unsqueeze(0).unsqueeze(0)

train_step = 0
val_step = 0
for e in range(cfg.epochs):
    train_loss = 0
    val_loss = 0
    pbar_update = cfg.batch_size
    model.train()
    with tqdm(total=len(train_set)) as pbar:
        for batch_idx, batch_data in enumerate(train_loader):
            if cfg.evaluate:
                break

            # get batch data
            bmode, gt_points, frame, gt_samples, pts_pala = batch_data
            frame = frame.to(cfg.device)
            gt_samples = gt_samples.to(cfg.device)

            # flatten channel and batch dimension
            frame = frame[:, wv_idx].flatten(0, 1).unsqueeze(1)
            gt_samples = gt_samples[:, wv_idx].flatten(0, 1)
            gt_samples[(gt_samples<=0) | (torch.isnan(gt_samples))] = 0 #torch.nan
            gt_true = torch.round(gt_samples.clone().unsqueeze(1)*cfg.upsample_factor).long()

            # inference
            masks_pred = model(frame)

            # train loss
            if cfg.model.lower() in ('stofnet', 'sincnet'):
                masks_true = samples2mask(gt_true, masks_pred) * cfg.mask_amplitude
                masks_true_blur = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                masks_pred_blur = F.conv1d(masks_pred, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                loss = loss_mse(masks_pred_blur.squeeze(1), masks_true_blur.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
            elif cfg.model.lower() == 'zonzini':
                # pick first ToA sample or maximum echo (Zonzini's model detect a single echo)
                gt_true //= cfg.upsample_factor
                max_values = torch.gather(abs(hilbert_transform(frame)), -1, gt_true)
                gt_true[gt_true==0] = 1e12
                idx_values = torch.argmin(gt_true, dim=-1) if True else max_values.argmax(-1)
                masks_true = torch.gather(gt_samples, -1, idx_values)
                loss = loss_mse(masks_pred, masks_true)
            train_loss += loss.item()
            train_step += 1

            # back-propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get estimated samples
            if cfg.model.lower() in ('stofnet', 'sincnet'):
                masks_supp = masks_pred.clone().detach()
                es_samples = samples2coords(masks_supp, window_size=cfg.kernel_size, threshold=cfg.th, upsample_factor=cfg.upsample_factor)
            elif cfg.model.lower() == 'zonzini':
                ideal_threshold = 0
                es_samples = masks_pred.clone().detach()

            if cfg.logging:
                wb.log({
                    'train_loss': loss.item(),
                    'train_step': train_step,
                    'train_points': (masks_true>0).sum(),
                })

            if cfg.logging and batch_idx%800 == 50:
                # convert mask to samples
                frame = unravel_batch_dim(frame)
                gt_samples = unravel_batch_dim(gt_samples)
                es_samples = unravel_batch_dim(es_samples)
                masks_pred = unravel_batch_dim(masks_pred)
                masks_true = unravel_batch_dim(masks_true)

                # channel plot
                fig = plot_channel_overview(frame[0].squeeze().cpu().numpy(), gt_samples[0].squeeze().cpu().numpy(), echoes=es_samples[0].cpu().numpy(), magnify_adjacent=True)
                wb_img_upload(fig, log_key='train_channels')
                
                if cfg.model.lower() in ('stofnet', 'sincnet'):
                    # image frame plot
                    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                    axs[0].imshow(masks_pred.flatten(0, 1).squeeze().detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                    axs[1].imshow(masks_true.flatten(0, 1).squeeze().detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                    plt.tight_layout()
                    wb_img_upload(fig, log_key='train_frames')
                    plt.close('all')

            pbar.update(pbar_update)

    train_loss = train_loss / len(train_set)

    if cfg.logging:
        wb.log({
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': e,
        })

    if not cfg.evaluate: scheduler.step()
    torch.cuda.empty_cache()

    # Validation
    model.eval()
    with tqdm(total=len(val_set)) as pbar:
        for batch_idx, batch_data in enumerate(val_loader):
            with torch.no_grad():

                # get batch data
                bmode, gt_points, frame, gt_samples, pts_pala = batch_data
                frame = frame.to(cfg.device)
                gt_samples = gt_samples.to(cfg.device)

                # flatten channel and batch dimension
                frame = frame[:, wv_idx].flatten(0, 1).unsqueeze(1)
                gt_samples = gt_samples[:, wv_idx].flatten(0, 1)
                gt_samples[(gt_samples<=0) | (torch.isnan(gt_samples))] = 0 #torch.nan
                gt_true = torch.round(gt_samples.clone().unsqueeze(1)*cfg.upsample_factor).long()

                # inference
                masks_pred = model(frame)

                # validation loss
                if cfg.model.lower() in ('stofnet', 'sincnet'):
                    masks_true = samples2mask(gt_true, masks_pred) * cfg.mask_amplitude
                    masks_true_blur = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                    masks_pred_blur = F.conv1d(masks_pred, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                    loss = loss_mse(masks_pred_blur.squeeze(1), masks_true_blur.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
                elif cfg.model.lower() == 'zonzini':
                    # pick first ToA sample or maximum echo (Zonzini's model detect a single echo)
                    gt_true //= cfg.upsample_factor
                    gt_true[gt_true==0] = 1e12
                    max_values = torch.gather(abs(hilbert_transform(frame)), -1, gt_true)
                    idx_values = torch.argmin(gt_true, dim=-1) if True else max_values.argmax(-1)
                    masks_true = torch.gather(gt_samples, -1, idx_values)
                    loss = loss_mse(masks_pred, masks_true)
                val_loss += loss.item()
                val_step += 1

                if cfg.model.lower() in ('stofnet', 'sincnet'):
                    # estimate ideal threshold
                    max_val = float(masks_true.max())
                    ideal_threshold = find_threshold(masks_pred.cpu()/max_val, masks_true.cpu()/max_val, window_size=cfg.kernel_size) * max_val

                    # get estimated samples
                    masks_supp = masks_pred.clone().detach()
                    es_samples = samples2coords(masks_supp, window_size=cfg.kernel_size, threshold=cfg.th, upsample_factor=cfg.upsample_factor)
                elif cfg.model.lower() == 'zonzini':
                    ideal_threshold = 0
                    es_samples = masks_pred.clone().detach()

                # get errors
                toa_errs = toa_rmse(gt_samples, es_samples, tol=cfg.etol)

                if cfg.logging:
                    wb.log({
                        'val_loss': loss.item(),
                        'val_step': val_step,
                        'val_points': (masks_true>0).sum(),
                        'val_toa_distance': torch.nanmean(toa_errs[0]),
                        'val_toa_precision': torch.nanmean(toa_errs[1]),
                        'val_toa_recall': torch.nanmean(toa_errs[2]),
                        'val_toa_jaccard': torch.nanmean(toa_errs[3]),
                        'val_toa_true_positive': torch.nanmean(toa_errs[4]),
                        'val_toa_false_positive': torch.nanmean(toa_errs[5]),
                        'val_toa_false_negative': torch.nanmean(toa_errs[6]),
                        'val_ideal_threshold': ideal_threshold,
                    })

                if cfg.logging and batch_idx%800 == 50:
                    # convert mask to samples
                    frame = unravel_batch_dim(frame)
                    gt_samples = unravel_batch_dim(gt_samples)
                    es_samples = unravel_batch_dim(es_samples)
                    masks_pred = unravel_batch_dim(masks_pred)
                    masks_true = unravel_batch_dim(masks_true)
                    # channel plot
                    fig = plot_channel_overview(frame[0].squeeze().cpu().numpy(), gt_samples[0].squeeze().cpu().numpy(), echoes=es_samples[0].cpu().numpy(), magnify_adjacent=True)
                    wb_img_upload(fig, log_key='val_channels')

                    if cfg.model.lower() in ('stofnet', 'sincnet'):
                        # image frame plot
                        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                        axs[0].imshow(masks_pred.flatten(0, 1).squeeze().detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                        axs[1].imshow(masks_true.flatten(0, 1).squeeze().detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                        plt.tight_layout()
                        wb_img_upload(fig, log_key='val_frames')
                        plt.close('all')

                pbar.update(pbar_update)

    torch.cuda.empty_cache()

    # save the model
    if cfg.logging:
        ckpt_path = script_path / 'ckpts' / (wb.name+'_epoch_'+str(e+1)+'.pth')
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
