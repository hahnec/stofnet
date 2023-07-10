import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from omegaconf import OmegaConf
from scipy.interpolate import interp1d


def upscale_1d(data, rescale_factor, fs=1):

    data_len = data.shape[0]
    x = np.linspace(0, data_len/fs, num=data_len, endpoint=True)
    t = np.linspace(0, data_len/fs, num=int(data_len*rescale_factor), endpoint=True)
    y = interp1d(x, data, axis=0)(t)

    return y


class ChirpDataset(Dataset):
    def __init__(self, root_dir, split_dirname='test', rf_scale_factor=10, transforms=None):
        
        # pass inputs to member variables
        self.root_dir = Path(root_dir)
        self.split_dirname = split_dirname
        self.rf_scale_factor = rf_scale_factor
        self.transforms = transforms

        # load sensor config
        self.cfg = OmegaConf.load(str(self.root_dir / 'sensor_specs.yaml'))
        self.cfg.speed_of_sound = 331.4 + 0.6 * self.cfg.temperature_celsius

        # initialize filename lists
        self.samples_env = []
        self.samples_iq = []
        self.gt_iq = []
        self.gt_positions = []
        self.labels = []

        # iterate through target classes
        target_dirs = [el for el in self.root_dir.iterdir() if el.is_dir()]
        for target_dir in target_dirs:

            # load sample paths
            samples_env, samples_iq = self._get_file_paths(str(target_dir / self.split_dirname))
            self.samples_env.extend(samples_env)
            self.samples_iq.extend(samples_iq)

            # load ground truth paths
            gt_env, gt_iq = self._get_file_paths(str(target_dir / 'truth'))

            # load ground truth positions
            gt_positions = np.genfromtxt(str(target_dir / 'truth' / 'true_measurement_positions.csv'), delimiter=',')[:, 1]

            # replicate ground truth data to match sample number
            gt_scale = len(samples_iq) // len(gt_iq)
            self.gt_iq.extend([gt for el in gt_iq for gt in [el,]*gt_scale])
            self.gt_positions.extend([gt for el in gt_positions for gt in [el,]*gt_scale])
            self.labels.extend([target_dir.name,] * len(samples_iq))
            
            assert len(self.samples_env) == len(self.samples_iq) == len(self.gt_iq) == len(self.gt_positions) == len(self.labels), \
                'inconsistent sample numbers'

    @staticmethod
    def _get_file_paths(dir_path):
        
        paths_iq = []
        paths_env = []
        seq_paths = [dir_path for dir_path in Path(dir_path).iterdir() if dir_path.is_dir()]
        for subdir_path in natsorted(seq_paths):
            for subfile in natsorted(Path(subdir_path).iterdir()):
                if subfile.name.__contains__('envelope'):
                    paths_env.append(subfile)
                elif subfile.name.__contains__('iq'):
                    paths_iq.append(subfile)

        return paths_env, paths_iq

    @staticmethod
    def iq2rf(iq_data, fc, fs, rescale_factor=1):
        
        # upscale IQ for RF representation
        data_len = iq_data.shape[0]
        x = np.linspace(0, data_len/fs, num=data_len, endpoint=True)
        t = np.linspace(0, data_len/fs, num=int(data_len*rescale_factor), endpoint=True)
        y = interp1d(x, iq_data, axis=0)(t)

        # IQ to RF conversion
        rf_data = y * np.exp(2j*np.pi*fc*t)

        return rf_data.real

    def get_channel_num(self):
        return 1

    def get_sample_num(self):
        return len(np.loadtxt(self.gt_iq[0]))

    def __len__(self):
        return len(self.gt_positions)

    def __getitem__(self, idx):

        # load data
        envelope_data = np.loadtxt(self.samples_env[idx])
        iq_data = np.loadtxt(self.samples_iq[idx])
        iq_gt = np.loadtxt(self.gt_iq[idx])
        gt_position = self.gt_positions[idx]
        label = self.labels[idx]

        # convert distance to travel time and sample index (GT position is [mm])
        toa = 2*(gt_position*1e-3) / self.cfg.speed_of_sound
        gt_sample = toa * self.cfg.fhz_sample * self.rf_scale_factor

        # convert to complex numbers
        iq_data = iq_data[:, 0] + 1j * iq_data[:, 1]
        iq_gt = iq_gt[:, 0] + 1j * iq_gt[:, 1]

        # convert to radio-frequency signal
        rf_data = self.iq2rf(iq_data, fc=self.cfg.fhz_carrier, fs=self.cfg.fhz_sample, rescale_factor=self.rf_scale_factor)
        rf_gt = self.iq2rf(iq_gt, fc=self.cfg.fhz_carrier, fs=self.cfg.fhz_sample, rescale_factor=self.rf_scale_factor)
        envelope_data = upscale_1d(envelope_data, rescale_factor=self.rf_scale_factor)

        if self.transforms:
            for transform in self.transforms:
                (envelope_data, _), (rf_data, gt_sample), (rf_gt, _) = [transform(data, sample)[:2] for data, sample in zip([envelope_data, rf_data, rf_gt], [gt_sample]*3)]

        return envelope_data, rf_data, rf_gt, gt_sample, gt_position, label


if __name__ == '__main__':

    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(3008)
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.zip_extract import zip_extract

    script_path = Path(__file__).parent.resolve()
    data_path = script_path / 'stof_chirp101_dataset'
    zip_extract(data_path)

    dataset = ChirpDataset(script_path / 'stof_chirp101_dataset', 'test')
    loader_args = dict(batch_size=2, num_workers=1, pin_memory=False)
    data_loader = DataLoader(dataset, shuffle=True, **loader_args)

    for batch_idx, batch_data in enumerate(data_loader):

        envelope_data, rf_data, rf_gt, gt_sample, gt_position, label = batch_data

        fs = dataset.cfg.fhz_sample
        rf_scale_factor = dataset.rf_scale_factor
        data_len1 = envelope_data.shape[-1]
        data_len2 = rf_gt.shape[-1]
        x = np.linspace(0, data_len1/fs, num=data_len1, endpoint=True)
        t = np.linspace(0, data_len2/fs/rf_scale_factor, num=data_len2, endpoint=True)

        plt.plot(t, rf_data[0], label='RF data')
        plt.plot(t, rf_data[0], '.', label='RF data points')
        plt.plot(t, rf_gt[0], label='RF ground truth')
        plt.plot(x, envelope_data[0], label='Envelope measurement')
        plt.plot([t[(gt_sample[0]).round().int()],]*2, [-.8*rf_data[0].max(), .8*rf_data[0].max()], linestyle='dashed', label='GT position')
        plt.title(label=label[0]+' @ '+str(float(gt_position[0]))+'mm')
        plt.legend()
        plt.show()