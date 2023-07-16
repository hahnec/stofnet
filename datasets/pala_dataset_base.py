import torch
from torch.utils.data import Dataset
from pathlib import Path
from natsort import natsorted
import numpy as np
import scipy.io
from scipy.interpolate import interp1d, interp2d
from scipy import signal
from typing import Union, List, Tuple
from omegaconf import OmegaConf

from utils.iq_conversion import iq2mp, mp2iq
from utils.svd_filter import svd_filter
from utils.beamform import bf_das
from utils.dict_dot import convert_to_dot_notation

mat2dict = lambda mat: dict([(k[0], v.squeeze()) for v, k in zip(mat[0][0], list(mat.dtype.descr))])
bf_demod_100_bw2iq = lambda rf_100bw: rf_100bw[:, 0::2, ...] - 1j*rf_100bw[:, 1::2, ...]

class PalaDatasetBase(Dataset):
    
    def __init__(
            self, 
            dataset_path = '', 
            sequences: Union[List, Tuple] = None,
            transforms = None,
            temporal_filter_opt: bool = False,
            compound_opt: bool = False,
            clutter_db: float = None,
            das_b4_temporal: bool = False,
            ):

        torch.manual_seed(3008)

        self.dataset_path = Path(dataset_path) / 'IQ'
        self.rfiq_key = 'IQ'

        self.transforms = transforms
        self.sequences = [0] if sequences is None else sequences
        self.temporal_filter_opt = temporal_filter_opt
        self.compound_opt = compound_opt
        self.clutter_db = 0 if clutter_db is None else clutter_db
        self.das_b4_temporal = das_b4_temporal

    def get_key(self, key=''):
        return self.all_metads[0][key]
    
    def get_rx_positions(self):
        x_positions = self.all_metads[0]['xe']
        return np.stack([x_positions, np.zeros_like(x_positions)]).T

    def compose_config(self):

        # create outside metadata object
        args_dict = {"rescale_factor": self.rescale_factor, "ch_gap": self.ch_gap}
        meta_dict = {k: float(v) for k, v in self.all_metads[0].items() if not isinstance(v, np.ndarray)}
        cfg = OmegaConf.create({**args_dict, **meta_dict})

        return cfg

    @staticmethod
    def img_norm(img):
        return (img-img.min())/(img.max()-img.min())

    def read_data(self):

        self.seqns_filenames = natsorted([str(fname.name) for fname in self.dataset_path.iterdir() if str(fname.name).lower().endswith('.mat')])

        seq_mat_name = 'Rat18_2D_PALA_0323_163558_sequence.mat' if str(self.dataset_path).lower().__contains__('rat') else 'PALA_InSilicoFlow_sequence.mat'
        seq_mat = scipy.io.loadmat(str(self.dataset_path.parent / seq_mat_name))

        frames_list = []
        labels_list = []
        mdatas_list = []
        pala_p_list = []
        for i in self.sequences:
            seq_frames, seq_labels, seq_mdatas, pts_pala = self.read_sequence(i, seq_mat)

            # sampling frequency (100% bandwidth mode of Verasonics) [Hz]
            seq_mdatas['fs'] = mat2dict(seq_mat['Receive'])['demodFrequency'] * 1e6

            # time between the emission and the beginning of reception [s]
            seq_mdatas['t0'] = 2*seq_mdatas['startDepth']*seq_mdatas['wavelength'] / seq_mdatas['c'] - mat2dict(seq_mat['TW'])['peak'] / seq_mdatas['f0']
            
            # collect sampling grid data
            p_data = mat2dict(seq_mat['PData']) if 'PData' in seq_mat.keys() else mat2dict(seq_mat['P'])
            seq_mdatas['Origin'] = p_data['Origin'] if 'Origin' in p_data.keys() else np.array([-72,   0,  16])
            seq_mdatas['Size'] = p_data['Size'] if 'Size' in p_data.keys() else np.array([ 84, 143,   1])
            seq_mdatas['PDelta'] = p_data['PDelta'] if 'PDelta' in p_data.keys() else np.array([1, 0, 1])

            seq_mdatas['param_x'] = (seq_mdatas['Origin'][0]+np.arange(seq_mdatas['Size'][1])*seq_mdatas['PDelta'][2])*seq_mdatas['wavelength']
            seq_mdatas['param_z'] = (seq_mdatas['Origin'][2]+np.arange(seq_mdatas['Size'][0])*seq_mdatas['PDelta'][0])*seq_mdatas['wavelength']

            # skip far end
            if False:
                if str(self.dataset_path).lower().__contains__('rat'):
                    seq_frames = seq_frames[:, :240, ...]
                else:
                    seq_frames = seq_frames[:, :192, ...]

            if self.temporal_filter_opt:
                if self.rfiq_key=='IQ':
                    seq_frames = svd_filter(seq_frames, lo_cut=4)
                    seq_frames = self.butterworth(seq_frames, seq_mdatas, frame_rate=float(seq_mdatas['FrameRate']))
                elif self.rfiq_key.__contains__('R'):
                    # DAS beamforming prior to temporal filtering
                    if self.das_b4_temporal:
                        bmodes = []
                        for frame_idx in range(seq_frames.shape[-1]):
                            bmode = bf_das(seq_frames[..., frame_idx], convert_to_dot_notation(seq_mdatas), compound_opt=False)
                            bmodes.append(self.img_norm(self.img_norm(bmode)**8))
                        seq_frames = np.array(bmodes).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)
                    # temporal filtering; loop over all angles
                    wv_idcs = range(seq_frames.shape[0]) if self.compound_opt else [seq_frames.shape[0]//2]
                    for wv_idx in wv_idcs:
                        seq_frames[wv_idx, ...] = svd_filter(seq_frames[wv_idx, ...], lo_cut=4, hi_cut=-1)
                        seq_frames[wv_idx, ...] = self.butterworth(seq_frames[wv_idx, ...], seq_mdatas, frame_rate=float(seq_mdatas['FrameRate']))
                seq_frames = seq_frames.swapaxes(0, -1).swapaxes(-2, -1)
                if self.rfiq_key == 'IQ':
                    seq_frames = abs(seq_frames)
                elif self.rfiq_key.__contains__('R'): 
                    seq_frames = seq_frames.swapaxes(1, 2)
            else:
                seq_frames = seq_frames.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)

            frames_list.append(seq_frames)
            labels_list.append(seq_labels)
            mdatas_list.append(seq_mdatas)
            pala_p_list.extend(pts_pala)
        
        # stack frames from different sequences
        self.all_frames = np.vstack(frames_list)
        self.all_labels = np.concatenate(labels_list, axis=0)
        self.all_metads = mdatas_list
        self.all_pala_p = pala_p_list

    def read_sequence(self, idx: int = 0, seq_mat=None):

        rf_mat = scipy.io.loadmat(self.dataset_path / self.seqns_filenames[idx])
        seq_mat = rf_mat if seq_mat is None else seq_mat

        seq_frames = rf_mat[self.rfiq_key].swapaxes(0, -1).swapaxes(1, -1)
        if 'ListPos' in rf_mat.keys():
            # insilico
            seq_labels = rf_mat['ListPos'].swapaxes(0, -1)
        elif (self.dataset_path.parent / 'Tracks').exists():
            # invivo (spots as pseudo ground-truth)
            path = sorted((self.dataset_path.parent / 'Tracks').iterdir())[idx]
            track_mat = scipy.io.loadmat(path)
            all_pts = np.vstack(track_mat['Track_raw'][0, 0][:, 0])
            seq_labels = [all_pts[i+1==all_pts[:, 2], :2] for i in range(int(max(all_pts[:, 2])))]
        seq_mdatas = mat2dict(seq_mat['P'])
        if self.rfiq_key == 'IQ':
            seq_frames = seq_frames.swapaxes(0, 1).swapaxes(1, 2)
        elif self.rfiq_key.__contains__('R'):
            if str(self.dataset_path).lower().__contains__('rat'):
                seq_frames = seq_frames.swapaxes(0, 1).reshape(-1, 800, 128, order='F').swapaxes(-2, -1).reshape(3, 640, 128, 800)
            else:
                seq_frames = seq_frames.reshape(1000, 3, -1, 128).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)
            seq_frames = bf_demod_100_bw2iq(seq_frames)

        # speed of sound [m/s]
        seq_mdatas['c'] = float(mat2dict(seq_mat['Resource'])['Parameters']['speedOfSound'])

        #central frequency [Hz]
        seq_mdatas['f0'] = mat2dict(seq_mat['Trans'])['frequency'] * 1e6

        # Wavelength [m]
        seq_mdatas['wavelength'] = seq_mdatas['c'] / seq_mdatas['f0']
     
        # x coordinates of transducer elements [m]
        seq_mdatas['xe'] = mat2dict(seq_mat['Trans'])['ElementPos'][:, 0]/1000

        # channel number
        seq_mdatas['Nelements'] = mat2dict(seq_mat['Trans'])['numelements']

        tx_steer = mat2dict(seq_mat['TX'])['Steer']
        seq_mdatas['angles_list'] = np.array([tx_steer*1, tx_steer*0, tx_steer*-1, tx_steer*0])
        seq_mdatas['angles_list'] = seq_mdatas['Angles'] if 'Angles' in seq_mdatas.keys() else seq_mdatas['angles_list'][:seq_mdatas['numTx'], 0]
        seq_mdatas['numTx'] = len(seq_mdatas['angles_list'])

        # load PALA tracks
        if str(self.dataset_path).lower().__contains__('rat'):
            fname_pala = Path('Tracks') / ('PALA_InVivoRatBrain_Tracks'+str(idx).zfill(3)+'.mat')
            mat_pala = scipy.io.loadmat(str(self.dataset_path.parent / fname_pala))
            pts_pala = np.vstack(mat_pala['Track_raw'][0, 0][:, 0])
            pts_list = [pts_pala[pts_pala[:, 2] == i+1].T for i in range(seq_frames.shape[-1])]
        else:
            fname_pala = Path('Tracks') / ('PALA_InSilico_Tracks'+str(idx).zfill(3)+'.mat')
            mat_pala = scipy.io.loadmat(str(self.dataset_path.parent / fname_pala))
            pala_local_methods = [el[0] for el in mat_pala['listAlgo'][0]]
            pala_local_results = {m: arr for arr, m in zip(mat_pala['Track_raw'][0], pala_local_methods)}
            pala_method = pala_local_methods[-1]
            pts_pala = pala_local_results[pala_method]
            pts_list = [pts_pala[pts_pala[:, 3] == i+1][:, 1:3][:, ::-1].T.copy() for i in range(seq_frames.shape[-1])]    # intensity max, zpos, xpos, frame index

        return seq_frames, seq_labels, seq_mdatas, pts_list

    def butterworth(self, frames, mdata, frame_rate=1000):

        # SVD filter only central plane wave
        but_b, but_a = signal.butter(2, np.array([frame_rate/20, frame_rate/4])/(frame_rate/2), btype='bandpass')
        #frames = signal.filtfilt(but_b, but_a, frames, axis=-1) # yields wrong results due to lagging?
        frames = signal.lfilter(but_b, but_a, frames, axis=-1)

        return frames

    def __getitem__(self, idx):

        # load data frame
        bmode_frame, metadata = (self.all_frames[idx], self.all_metads[0])

        # add noise according to PALA study
        if np.isreal(self.clutter_db) and self.clutter_db < 0:
            bmode_frame = add_pala_noise(bmode_frame, clutter_db=self.clutter_db)

        # convert label data to ground-truth representation(s)
        if str(self.dataset_path).lower().__contains__('insilico'):
            label_raw = self.all_labels[idx] if len(self.all_labels) > 1 else None
            nan_idcs = np.isnan(label_raw[0]) & np.isnan(label_raw[2])
            gt_points = label_raw[:, ~nan_idcs] * metadata['wavelength']

        elif (self.dataset_path.parent / 'Tracks').exists():
            gt_points = self.all_labels[0][idx].T

        else:
            gt_points = np.array([])

        bmode_frame = self.transforms(bmode_frame)

        return bmode_frame, gt_points

    def __len__(self):
        return len(self.all_frames)
