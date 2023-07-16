import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm
import random
import os
from omegaconf import OmegaConf
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(str(Path(__file__).parent / "stofnet"))
sys.path.append(str(Path(__file__).parent.parent))

from models import StofNet, ZonziniNetLarge, ZonziniNetSmall, SincNet, GradPeak, Kuleshov, EDSR_1D, ESPCN_1D
from datasets.pala_dataset_rf import PalaDatasetRf
from datasets.chirp_dataset import ChirpDataset
from utils.mask2samples import coords2mask, mask2nested_list, mask2coords
from utils.gaussian import gaussian_kernel
from utils.hilbert import hilbert_transform
from utils.metrics import toa_rmse
from utils.threshold import find_threshold
from utils.plotting import wb_img_upload, plot_channel_overview
from utils.transforms import NormalizeVol, CropChannelData, AddNoise
from utils.collate_fn import collate_fn
from utils.zip_extract import zip_extract
from utils.early_stop import EarlyStopping

# load config
script_path = Path(__file__).parent.resolve()
cfg = OmegaConf.load(str(script_path / 'config.yaml'))

# override loaded config file with CLI arguments
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

unravel_batch_dim = lambda x: x.reshape(cfg.batch_size, x.shape[0]//cfg.batch_size, -1)

# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

# load dataset
transforms_list = [NormalizeVol()]
if cfg.data_dir.lower().__contains__('pala') or cfg.data_dir.lower().__contains__('rat'):
    # load dataset
    if not cfg.evaluate: transforms_list += [AddNoise(snr=cfg.snr_db)]
    dataset = PalaDatasetRf(
        dataset_path = cfg.data_dir,
        sequences = cfg.sequences,
        rescale_factor = cfg.rf_scale_factor,
        ch_gap = cfg.ch_gap,
        angle_threshold = cfg.angle_threshold,
        clutter_db = cfg.clutter_db,
        temporal_filter_opt=cfg.temporal_filter,
        pow_law_opt = cfg.pow_law_opt,
        transforms = torch.nn.Sequential(*transforms_list),
        )
    
    # data-related config
    angles_list = dataset.get_key('angles_list')
    wv_idcs = range(len(angles_list))
    wv_idx = 1
    cfg.fs = float(dataset.get_key('fs'))
    cfg.c = float(dataset.get_key('c'))
    cfg.wavelength = float(dataset.get_key('wavelength'))

elif cfg.data_dir.lower().__contains__('chirp'):
    # extract data folder from zip
    data_path = script_path / cfg.data_dir
    zip_extract(data_path)
    # load dataset
    if not cfg.evaluate: transforms_list += [CropChannelData(ratio=cfg.crop_ratio, resize=False), AddNoise(snr=cfg.snr_db)]
    dataset = ChirpDataset(
        root_dir = data_path,
        split_dirname = 'test' if cfg.evaluate else 'train',
        rf_scale_factor = cfg.rf_scale_factor,
        transforms = torch.nn.Sequential(*transforms_list),
    )
    # data-related config
    cfg.fs = dataset.cfg.fhz_sample
    cfg.c = dataset.cfg.speed_of_sound
    # override collate function
    collate_fn = None
else:
    raise Exception('No dataset class found for given data path')

channel_num = dataset.get_channel_num()
sample_num = dataset.get_sample_num()

# split into train / validation partitions
val_percent = 1 if cfg.evaluate else 0.2
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

# create data loaders
num_workers = min(4, os.cpu_count())
pin_memory = True if cfg.device == "cuda" else False
loader_args = dict(batch_size=cfg.batch_size, num_workers=num_workers, pin_memory=pin_memory)
train_loader = DataLoader(train_set, collate_fn=collate_fn, shuffle=True, **loader_args) if not cfg.evaluate else None
val_loader = DataLoader(val_set, collate_fn=collate_fn, shuffle=False, drop_last=True, **loader_args)

# instantiate logging
if cfg.logging:
    wb = wandb.init(project='StofNet', resume='allow', anonymous='must', config=cfg, group=cfg.logging)
    wb.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr, val_percent=val_percent))
    wandb.define_metric('train_loss', step_metric='train_step')
    wandb.define_metric('train_points', step_metric='train_step')
    wandb.define_metric('val_loss', step_metric='val_step')
    wandb.define_metric('val_points', step_metric='val_step')
    wandb.define_metric('val_ideal_threshold', step_metric='val_step')
    wandb.define_metric('inference_time', step_metric='val_step')
    wandb.define_metric('val_toa_distance', step_metric='val_idx')
    wandb.define_metric('val_toa_precision', step_metric='val_idx')
    wandb.define_metric('val_toa_recall', step_metric='val_idx')
    wandb.define_metric('val_toa_jaccard', step_metric='val_idx')
    wandb.define_metric('val_toa_true_positive', step_metric='val_idx')
    wandb.define_metric('val_toa_false_positive', step_metric='val_idx')
    wandb.define_metric('val_toa_false_negative', step_metric='val_idx')
    wandb.define_metric('lr', step_metric='epoch')

# load model
if cfg.model.lower() == 'stofnet':
    model = StofNet(upsample_factor=cfg.upsample_factor)
elif cfg.model.lower() == 'zonzini':
    model = ZonziniNetSmall() if cfg.data_dir.lower().__contains__('chirp') else ZonziniNetLarge()
elif cfg.model.lower() == 'kuleshov':
    model = Kuleshov(input_length=sample_num*cfg.rf_scale_factor, output_length=sample_num*cfg.rf_scale_factor*cfg.upsample_factor)
elif cfg.model.lower() == 'edsr':
    model = EDSR_1D(num_channels=1, num_features=64, num_blocks=8, upscale_factor=cfg.upsample_factor)
elif cfg.model.lower() == 'espcn':
    model = ESPCN_1D(upscale_factor=cfg.upsample_factor)
elif cfg.model.lower() == 'sincnet':
    cfg.upsample_factor = 1
    sincnet_params = {'input_dim': sample_num*cfg.rf_scale_factor,
                        'fs': cfg.fs*cfg.rf_scale_factor,
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
elif cfg.model.lower() == 'gradpeak':
    # non-trainable gradient-based detection
    echo_max = 1 if cfg.data_dir.lower().__contains__('chirp') else float('inf')
    model = GradPeak(threshold=cfg.th, rescale_factor=cfg.rf_scale_factor, echo_max=echo_max, onset_opt=cfg.data_dir.lower().__contains__('chirp'))
    cfg.evaluate = True
else:
    raise Exception('Model not recognized')

model = model.to(cfg.device)
model.eval()

if not cfg.model.lower() == 'gradpeak':
    if cfg.model_file:
        ckpt_paths = [fn for fn in (script_path / 'ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        model.load_state_dict(state_dict)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
early_stopping = EarlyStopping(patience=cfg.patience, delta=cfg.delta)

# loss settings
loss_mse = nn.MSELoss(reduction='mean')
loss_l1 = nn.L1Loss(reduction='mean')
zero_l1 = torch.zeros((cfg.batch_size*channel_num, sample_num*cfg.rf_scale_factor*cfg.upsample_factor), device=cfg.device, dtype=torch.float32)
loss_l1_arg = lambda y: loss_l1(y, zero_l1)
gauss_kernel_1d = torch.tensor(gaussian_kernel(size=cfg.kernel_size, sigma=cfg.sigma), dtype=torch.float32, device=cfg.device).unsqueeze(0).unsqueeze(0)

# initialze metrics
total_inference_time = []
total_distance = []
total_jaccard = []

# iterate through epochs
epochs = 1 if cfg.evaluate else cfg.epochs
train_step, val_step = 0, 0
for e in range(epochs):
    if not cfg.evaluate:
        # train
        model.train()
        train_loss = 0
        with tqdm(total=len(train_set)) as pbar:
            for batch_idx, batch_data in enumerate(train_loader):

                # get batch data
                if cfg.data_dir.lower().__contains__('pala') or cfg.data_dir.lower().__contains__('rat'):
                    bmode, gt_points, frame, gt_sample, pts_pala = batch_data
                    frame = frame[:, wv_idx].flatten(0, 1).unsqueeze(1)
                    gt_sample = gt_sample[:, wv_idx].flatten(0, 1)
                elif cfg.data_dir.lower().__contains__('chirp'):
                    envelope_data, rf_data, rf_gt, gt_sample, gt_position, label = batch_data
                    frame = rf_data.float().unsqueeze(1)
                    gt_sample = gt_sample.unsqueeze(1)
                frame = frame.to(cfg.device)
                gt_sample = gt_sample.to(cfg.device)
                gt_sample[(gt_sample<=0) | (torch.isnan(gt_sample))] = 0 #torch.nan
                gt_true = torch.round(gt_sample.clone().unsqueeze(1)*cfg.upsample_factor).long()

                # inference
                masks_pred = model(frame)

                # train loss
                if cfg.model.lower() in ('stofnet', 'sincnet', 'kuleshov', 'edsr', 'espcn'):
                    # get estimated samples
                    es_sample = mask2coords(masks_pred, window_size=cfg.nms_win_size, threshold=cfg.th, upsample_factor=cfg.upsample_factor)
                    # loss computation
                    masks_true = coords2mask(gt_true, masks_pred)
                    masks_true_blur = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                    masks_true_blur /= masks_true_blur.max() 
                    masks_true_blur *= cfg.mask_amplitude
                    loss = loss_mse(masks_pred.squeeze(1), masks_true_blur.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
                elif cfg.model.lower() == 'zonzini':
                    # get estimated samples: pick first ToA sample or maximum echo (Zonzini's model detect a single echo)
                    es_sample = masks_pred.clone().detach()
                    gt_true //= cfg.upsample_factor
                    max_values = torch.gather(abs(hilbert_transform(frame)), -1, gt_true)
                    gt_true[gt_true==0] = 1e12
                    idx_values = torch.argmin(gt_true, dim=-1) if True else max_values.argmax(-1)
                    masks_true = torch.gather(gt_sample, -1, idx_values).float()
                    loss = loss_mse(masks_pred, masks_true)
                train_loss += loss.item()
                train_step += 1

                # back-propagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if cfg.logging:
                    wb.log({
                        'train_step': train_step,
                        'train_loss': loss.item(),
                        'train_points': (masks_true>0).sum(),
                    })

                if cfg.logging and batch_idx%800 == 50:
                    # unravel channel and batch dimension
                    frame = unravel_batch_dim(frame)
                    gt_sample = unravel_batch_dim(gt_sample)
                    es_sample = unravel_batch_dim(es_sample)
                    masks_pred = unravel_batch_dim(masks_pred)
                    masks_true = unravel_batch_dim(masks_true)

                    # channel plot
                    fig = plot_channel_overview(frame[0].cpu().numpy(), gt_sample[0].cpu().numpy(), echoes=es_sample[0].cpu().numpy(), magnify_adjacent=True if cfg.data_dir.lower().__contains__('pala') else False)
                    wb_img_upload(fig, log_key='train_channels')
                    
                    if cfg.model.lower() in ('stofnet', 'sincnet', 'kuleshov', 'edsr', 'espcn'):
                        # image frame plot
                        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                        axs[0].imshow(masks_pred.flatten(0, 1).detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                        axs[1].imshow(masks_true.flatten(0, 1).detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                        plt.tight_layout()
                        wb_img_upload(fig, log_key='train_frames')
                        plt.close('all')

                pbar.update(cfg.batch_size)

        train_loss = train_loss / len(train_set)

        if cfg.logging:
            wb.log({
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': e,
            })

        scheduler.step()
        torch.cuda.empty_cache()

    # validation or test
    model.eval()
    val_loss = 0
    with tqdm(total=len(val_set)) as pbar:
        for batch_idx, batch_data in enumerate(val_loader):
            with torch.no_grad():
                
                # get batch data
                if cfg.data_dir.lower().__contains__('pala') or cfg.data_dir.lower().__contains__('rat'):
                    bmode, gt_points, frame, gt_sample, pts_pala = batch_data
                    frame = frame[:, wv_idx].flatten(0, 1).unsqueeze(1)
                    gt_sample = gt_sample[:, wv_idx].flatten(0, 1)
                elif cfg.data_dir.lower().__contains__('chirp'):
                    envelope_data, rf_data, rf_gt, gt_sample, gt_position, label = batch_data
                    frame = rf_data.float().unsqueeze(1)
                    gt_sample = gt_sample.unsqueeze(1)
                frame = frame.to(cfg.device)
                gt_sample = gt_sample.to(cfg.device)
                gt_sample[(gt_sample<=0) | (torch.isnan(gt_sample))] = 0 #torch.nan
                gt_true = torch.round(gt_sample.clone().unsqueeze(1)*cfg.upsample_factor).long()

                # inference
                tic = time.process_time()
                masks_pred = model(frame)
                toc = time.process_time() - tic

                # validation loss
                if cfg.model.lower() in ('stofnet', 'sincnet', 'kuleshov', 'edsr', 'espcn'):
                    # get estimated samples
                    es_sample = mask2coords(masks_pred, window_size=cfg.nms_win_size, threshold=cfg.th, upsample_factor=cfg.upsample_factor)
                    # loss computation
                    masks_true = coords2mask(gt_true, masks_pred)
                    masks_true_blur = F.conv1d(masks_true, gauss_kernel_1d, padding=cfg.kernel_size // 2)
                    masks_true_blur /= masks_true_blur.max() 
                    masks_true_blur *= cfg.mask_amplitude
                    loss = loss_mse(masks_pred.squeeze(1), masks_true_blur.squeeze(1).float()) + loss_l1_arg(masks_pred.squeeze(1)) * cfg.lambda_value
                    val_loss += loss.item()
                    
                    # estimate ideal threshold
                    max_val = float(masks_true.max()) if float(masks_true.max()) != 0 else 1
                    ideal_th = find_threshold(masks_pred.cpu(), masks_true.cpu()/max_val, norm_opt=True) * max_val

                elif cfg.model.lower() in ('zonzini', 'gradpeak'):
                    # get estimated samples: pick first ToA sample or maximum echo (Zonzini's model detects single echoes)
                    es_sample = masks_pred.clone().detach()
                    # loss computation
                    gt_true //= cfg.upsample_factor
                    max_values = torch.gather(abs(hilbert_transform(frame)), -1, gt_true)
                    gt_true[gt_true==0] = 1e12
                    idx_values = torch.argmin(gt_true, dim=-1) if True else max_values.argmax(-1)
                    masks_true = torch.gather(gt_sample, -1, idx_values)
                    loss = loss_mse(masks_pred, masks_true)
                    val_loss += loss.item()
                    ideal_th = 0
                val_step += 1

                # get errors
                toa_errs = toa_rmse(gt_sample, es_sample, tol=cfg.etol)

                if cfg.logging:
                    wb.log({
                        'val_step': val_step,
                        'val_loss': loss.item(),
                        'val_ideal_threshold': ideal_th,
                        'inference_time': toc/cfg.batch_size,
                    })

                    # evaluation metrics
                    for k, toa_err in enumerate(toa_errs):
                        total_distance.append(float(toa_err[0]))
                        total_jaccard.append(float(toa_err[3]))
                        total_inference_time.append(toc/cfg.batch_size)
                        wb.log({
                            'val_idx': (val_step-1)*cfg.batch_size*channel_num + k,
                            'val_points': (masks_true>0).sum(),
                            'val_toa_distance': toa_err[0],
                            'val_toa_precision': toa_err[1],
                            'val_toa_recall': toa_err[2],
                            'val_toa_jaccard': toa_err[3],
                            'val_toa_true_positive': toa_err[4],
                            'val_toa_false_positive': toa_err[5],
                            'val_toa_false_negative': toa_err[6],
                        })

                    # skip every other frame
                    if batch_idx % 100 == 1 and cfg.evaluate:
                        # unravel channel and batch dimension
                        frame = unravel_batch_dim(frame)
                        gt_sample = unravel_batch_dim(gt_sample)
                        es_sample = unravel_batch_dim(es_sample)
                        masks_pred = unravel_batch_dim(masks_pred)
                        masks_true = unravel_batch_dim(masks_true)
                        # channel plot
                        fig = plot_channel_overview(frame[0].cpu().numpy(), gt_sample[0].cpu().numpy(), echoes=es_sample[0].cpu().numpy(), magnify_adjacent=True if cfg.data_dir.lower().__contains__('pala') else False)
                        wb_img_upload(fig, log_key='val_channels')
                        # channel plot artifact
                        frame_artifact = wandb.Artifact(f'frame_{str(batch_idx).zfill(5)}', type='data', description=cfg.model)
                        for key, var in zip(['data', 'toa', 'gt'], [frame, es_sample, gt_sample]):
                            table = wandb.Table(data=np.transpose(var.cpu().numpy(), axes=(1, 0, 2)), columns=['Column'+str(el) for el in np.arange(var.shape[0])])
                            frame_artifact.add(table, key)
                        wandb.log_artifact(frame_artifact)

                        if cfg.model.lower() in ('stofnet', 'sincnet', 'kuleshov', 'edsr', 'espcn'):
                            # image frame plot
                            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                            axs[0].imshow(masks_pred.flatten(0, 1).detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                            axs[1].imshow(masks_true.flatten(0, 1).detach().cpu().numpy()[:, 256:256+2*masks_pred.flatten(0, 1).shape[0]])
                            plt.tight_layout()
                            wb_img_upload(fig, log_key='val_frames')
                            plt.close('all')

                pbar.update(cfg.batch_size)

    torch.cuda.empty_cache()

    # early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Finished at epoch:", e)
        break

if cfg.logging:

    # wandb summary
    model_summary = summary(model)
    wandb.summary['model_name'] = cfg.model
    wandb.summary['total_parameters'] = int(str(model_summary).split('\n')[-3].split(' ')[-1].replace(',',''))
    wandb.summary['total_jaccard'] = np.nanmean(total_jaccard)
    wandb.summary['total_inference_time'] = np.nanmean(total_inference_time)
    wandb.summary['total_distance_mean'] = np.nanmean(total_distance)
    wandb.summary['total_distance_std'] = np.std(np.array(total_distance)[~np.isnan(total_distance)])

    # save the model
    if not cfg.evaluate:
        ckpt_path = script_path / 'ckpts' / (wb.name+'_rf-scale'+str(cfg.rf_scale_factor)+'_epoch_'+str(e+1)+'.pth')
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        ckpt_artifact = wandb.Artifact('checkpoint', type='model', description=cfg.model)
        ckpt_artifact.add_file(ckpt_path)
        wandb.log_artifact(ckpt_artifact)

    wandb.finish()
