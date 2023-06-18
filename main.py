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
import sys
sys.path.append(str(Path(__file__).parent / "stofnet"))
sys.path.append(str(Path(__file__).parent.parent))

from model import StofNet
from dataloaders.dataset_pala_rf import InSilicoDatasetRf
from utils.gaussian import gaussian_kernel
from utils.mask2samples import samples2mask, samples2nested_list
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
    wandb.define_metric('val_loss', step_metric='val_step')
    wandb.define_metric('train_toa_diff', step_metric='train_step')
    wandb.define_metric('train_points', step_metric='train_step')
    wandb.define_metric('val_toa_diff', step_metric='val_step')
    wandb.define_metric('lr', step_metric='epoch')

# load model
model = StofNet(upsample_factor=cfg.upsample_factor, hilbert_opt=cfg.hilbert_opt, concat_oscil=cfg.oscil_opt)
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
            gt_samples[gt_samples<=0] = torch.nan
            gt_true = torch.round(gt_samples.clone().unsqueeze(1)*cfg.upsample_factor).long()

            # inference
            masks_pred = model(frame)

            # train loss
            masks_true = samples2mask(gt_true, masks_pred) * 1
            masks_true = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
            masks_blur = F.conv1d(masks_pred, gauss_kernel_1d, padding=cfg.kernel_size // 2)
            loss = loss_mse(masks_blur.squeeze(1), masks_true.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
            train_loss += loss.item()
            train_step += 1

            # convert mask to samples
            frame = unravel_batch_dim(frame)
            gt_samples = unravel_batch_dim(gt_samples)
            masks_pred = unravel_batch_dim(masks_pred)
            masks_true = unravel_batch_dim(masks_true)
            es_samples = samples2nested_list(masks_pred, window_size=cfg.kernel_size, upsample_factor=cfg.upsample_factor)

            # back-propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cfg.logging:
                wb.log({
                    'train_loss': loss.item(),
                    'train_step': train_step,
                    #'train_toa_diff': toa_diff,
                    #'train_points': pts_train_num,
                })

            if cfg.logging and batch_idx%200 == 50:
                # channel plot
                fig = plot_channel_overview(frame[0].squeeze().cpu().numpy(), gt_samples[0].squeeze().cpu().numpy(), echoes=es_samples[0], magnify_adjacent=True)
                wb_img_upload(fig, log_key='train_channels')
                
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

    scheduler.step()
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
                gt_samples[gt_samples<=0] = torch.nan
                gt_true = torch.round(gt_samples.clone().unsqueeze(1)*cfg.upsample_factor).long()

                # inference
                masks_pred = model(frame)

                # validation loss
                masks_true = samples2mask(gt_true, masks_pred) * 1
                masks_true = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                masks_blur = F.conv1d(masks_pred, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                loss = loss_mse(masks_blur.squeeze(1), masks_true.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
                val_loss += loss.item()
                val_step += 1

                # convert mask to samples
                frame = unravel_batch_dim(frame)
                gt_samples = unravel_batch_dim(gt_samples)
                masks_pred = unravel_batch_dim(masks_pred)
                masks_true = unravel_batch_dim(masks_true)
                es_samples = samples2nested_list(masks_pred, window_size=cfg.kernel_size)

                if cfg.logging:
                    wb.log({
                        'val_loss': loss.item(),
                        'val_step': val_step,
                        #'val_toa_diff': toa_diff,
                        #'val_points': pts_train_num,
                    })

                if cfg.logging and batch_idx%200 == 50:
                    # channel plot
                    fig = plot_channel_overview(frame[0].squeeze().cpu().numpy(), gt_samples[0].squeeze().cpu().numpy(), echoes=es_samples[0], magnify_adjacent=True)
                    wb_img_upload(fig, log_key='val_channels')

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
