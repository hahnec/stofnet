import torch
from torch.utils.data import Dataset
from pathlib import Path
from natsort import natsorted
import numpy as np
import scipy.io
from scipy.interpolate import interp1d, interp2d
from typing import Union, List, Tuple
from omegaconf import OmegaConf
from skimage.feature import blob_log

from datasets.pala_dataset_base import PalaDatasetBase
from utils.pala_noise import add_pala_noise
from utils.pow_law import compensate_pow_law
from utils.beamform import bf_das
from utils.dict_dot import convert_to_dot_notation
from utils.centroids import weighted_avg, argmax_blob
from utils.centroids import regional_max
from utils.radial_pala import radial_pala


mat2dict = lambda mat: dict([(k[0], v.squeeze()) for v, k in zip(mat[0][0], list(mat.dtype.descr))])


class PalaDatasetRf(PalaDatasetBase):
    
    def __init__(
            self, 
            dataset_path = '', 
            sequences: Union[List, Tuple] = None,
            rescale_factor: float = None,
            ch_gap: int = 1,
            angle_threshold: float = None,
            clutter_db: float = None,
            temporal_filter_opt: bool = False,
            compound_opt: bool = False,
            pow_law_opt: bool = False,
            transforms = None,
            echo_max = None,
            das_b4_temporal = False,
            ):
        super(PalaDatasetRf, self).__init__(dataset_path, sequences, transforms, temporal_filter_opt, compound_opt, clutter_db, das_b4_temporal)

        self.dataset_path = Path(dataset_path) / 'RF'
        self.rescale_factor = 1 if rescale_factor is None else rescale_factor
        self.ch_gap = 1 if ch_gap is None else ch_gap
        self.rfiq_key = 'RFdata' if str(self.dataset_path).lower().__contains__('insilico') else 'RData'

        self.angle_threshold = angle_threshold
        self.pow_law_opt = False if pow_law_opt is None else pow_law_opt
        self.echo_max = 10 if echo_max is None else echo_max
        self.das_b4_temporal = das_b4_temporal

        self.read_data()

    @staticmethod
    def get_vsource_from_meta(mdata, tx_idx=0, beta=1e-8):

        # extent of the phased-array
        width = mdata['xe'][-1] - mdata['xe'][0]
        # virtual source (non-planar wave assumption)
        theta = mdata['angles_list'][tx_idx]
        vsource = [-width*np.cos(theta)*np.sin(theta)/beta, -width*np.cos(theta)**2/beta]

        return vsource, width

    def get_vsource(self, tx_idx=1, beta=1e-8):
        return self.get_vsource_from_meta(self.all_metads[0], tx_idx=tx_idx, beta=beta)

    def get_rx_positions(self):
        x_positions = self.all_metads[0]['xe'][::self.ch_gap]
        return np.stack([x_positions, np.zeros_like(x_positions)]).T

    def get_channel_num(self):
        return self.all_frames[0].shape[-1] // self.ch_gap

    def get_sample_num(self):
        return self.all_frames[0].shape[1]

    @staticmethod
    def batched_iq2rf(iq_data, mod_freq, rescale_factor=1):

        data_len = iq_data.shape[1]

        x = np.linspace(0, data_len/mod_freq, num=data_len, endpoint=True)
        t = np.linspace(0, data_len/mod_freq, num=int(data_len*rescale_factor), endpoint=True)
        
        f = interp1d(x, iq_data, axis=1)
        y = f(t)

        rf_data = y * np.exp(2*1j*np.pi*mod_freq*t[:, None])

        rf_data = 2**.5 * rf_data.real

        return rf_data

    @staticmethod
    def project_points_toa(points, metadata, vsource, width, xe, angle_threshold=None):

        # find transmit travel distances considering virtual source
        nonplanar_tdx = np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])   # switched half-width shifted vsource position norm
        virtual_tdxs = np.hypot(points[0, ...]-vsource[0], points[1, ...]-vsource[1])   # vector norm of point to vsource
        dtxs = np.repeat((virtual_tdxs - nonplanar_tdx)[:, None], xe.size, axis=1)

        # find receive travel distances
        x_diff = points[0, ...][:, None] - np.repeat(xe[None, :], points[0, ...].shape[0], axis=0)
        drxs = np.hypot(x_diff, points[1, ...][:, None])

        # convert overall travel distances to travel times
        tau = (dtxs + drxs) / metadata['c']

        # convert travel times to sample indices (deducting time between the emission and the beginning of reception [s])
        sample_positions = (tau-metadata['t0']) * metadata['fs']

        if angle_threshold is not None:
            # reject projections at steep angles
            reception_angles = np.arctan(points[1, ...][:, None] / x_diff) * 180 / np.pi
            sample_positions[(90-abs(reception_angles))>angle_threshold] = float('NaN')

        return sample_positions

    def __getitem__(self, idx):

        wv_idx = 1

        # load data frame
        frame_raw, metadata = (self.all_frames[idx], self.all_metads[0])

        # pala points
        pts_pala = self.all_pala_p[idx]

        # beamforming
        if not self.das_b4_temporal:
            bmode_frame = bf_das(frame_raw, convert_to_dot_notation(metadata), compound_opt=False)
            bmode_frame = bmode_frame[wv_idx, ...] if not self.compound_opt else bmode_frame.sum(0)
            bmode_frame = self.img_norm(bmode_frame) if False else self.img_norm(self.img_norm(bmode_frame)**4)
        else:
            bmode_frame = frame_raw[wv_idx, ...].copy()

        # prepare frame channels
        channel_frame = self.batched_iq2rf(frame_raw, mod_freq=metadata['fs'], rescale_factor=self.rescale_factor)
        channel_frame = channel_frame.swapaxes(-2, -1)[:, ::self.ch_gap, ...]

        # power law compensation
        if self.pow_law_opt:
            #t = torch.arange(0, len(data_batch[:, 0])/param.fs/cfg.enlarge_factor, 1/param.fs/cfg.enlarge_factor, device=data_batch.device, dtype=data_batch.dtype)
            max_val = channel_frame.max() * 1.5
            channel_frame = compensate_pow_law(channel_frame, a=0.8888322820090128, b=0.4904105969203961, c=metadata['c'], fkHz=metadata['f0'], sample_rate=metadata['fs']*self.rescale_factor)
            channel_frame = channel_frame/channel_frame.max() * max_val
            bmode_frame = compensate_pow_law(bmode_frame.T, a=0.8888322820090128, b=0.4904105969203961, c=metadata['c'], fkHz=metadata['f0'], sample_rate=metadata['fs']*self.rescale_factor).T
            bmode_frame = self.img_norm(bmode_frame)
            
        # add noise according to PALA study
        if np.isreal(self.clutter_db) and self.clutter_db < 0:
            channel_frame = add_pala_noise(channel_frame, clutter_db=self.clutter_db)

        # convert label data to ground-truth representation(s)
        if str(self.dataset_path).lower().__contains__('insilico'):
            label_raw = self.all_labels[idx] if len(self.all_labels) > 1 else None
            nan_idcs = np.isnan(label_raw[0]) & np.isnan(label_raw[2])
            gt_points = label_raw[:, ~nan_idcs] * metadata['wavelength']

        elif False and (self.dataset_path.parent / 'Tracks').exists():
            gt_points = (self.all_labels[idx].T + metadata['Origin'][::2][:, None]) * metadata['wavelength']
            gt_points = np.repeat(gt_points, 2, 0)

        elif bmode_frame is not None:
            # caution! not accurate enough for phase precision
            #yxr_pts = blob_log(bmode_frame, max_sigma=4, threshold=.11)
            yxr_pts = regional_max(bmode_frame, point_num=self.echo_max)
            #yxr_pts[:, :2] = weighted_avg(bmode_frame, yxr_pts[:, :2], w=2)
            yxr_pts[:, :2] = argmax_blob(bmode_frame, yxr_pts[:, :2], w=4)
            yxr_pts[:, :2] = radial_pala(bmode_frame, yxr_pts[:, :2], w=2)
            gt_points = (yxr_pts[:, :2][:, ::-1].T + metadata['Origin'][::2][:, None]) * metadata['wavelength']
            gt_points = np.repeat(gt_points, 2, 0) # replicate for 4 dimensions

            s=gt_points[::2, :]/metadata['wavelength'] - metadata['Origin'][::2][:, None]
            #import matplotlib.pyplot as plt
            #plt.imshow(bmode_frame)
            #plt.plot(yxr_pts[:, 1], yxr_pts[:, 0], 'rx')
            #plt.plot(s[0, :], s[1, :], 'kx', label='GT')
            #plt.show()

        else:
            gt_points = np.array([])

        # GT sample projection
        if gt_points.size > 0:
            gt_samples = []
            # iterate over plane waves
            for tx_idx in range(metadata['numTx']):

                # virtual source
                vsource, width = self.get_vsource_from_meta(metadata, tx_idx=tx_idx)

                # project points to time-of-arrival
                sample_positions = self.project_points_toa(gt_points[::2, ...], metadata, vsource, width, metadata['xe'], angle_threshold=self.angle_threshold)

                # convert to time
                time_position = sample_positions / metadata['fs']

                sample_positions *= self.rescale_factor

                gt_samples.append(sample_positions)
            gt_samples = np.stack(gt_samples)
            gt_samples = gt_samples.swapaxes(-2, -1)[:, ::self.ch_gap, ...]

        else:
            gt_samples = np.array([])

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(channel_frame[1, 16, :])
        #plt.plot([gt_samples[1, 16],]*2, [300, -300])
        #plt.show()

        if self.transforms:
            for transform in self.transforms:
                bmode_frame = transform(bmode_frame)
                channel_frame[wv_idx, ...] = transform(channel_frame[wv_idx, ...])

        return bmode_frame, gt_points, channel_frame, gt_samples, pts_pala
    
    def __len__(self):
        return len(self.all_frames)
