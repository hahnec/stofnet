import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from omegaconf import OmegaConf
from scipy.interpolate import interp1d


class ChirpDataset(Dataset):
    def __init__(self, root_dir, split_dirname='test', rf_scale_factor=20):
        
        self.root_dir = Path(root_dir)
        self.split_dirname = split_dirname
        self.rf_scale_factor = rf_scale_factor

        # load sample paths
        self.samples_env, self.samples_iq = self._get_file_paths(str(self.root_dir / self.split_dirname))

        # load ground truth paths
        self.gt_env, self.gt_iq = self._get_file_paths(str(self.root_dir / 'truth'))

        # load ground truth positions
        self.gt_positions = np.genfromtxt(str(self.root_dir / 'truth' / 'true_measurement_positions.csv'), delimiter=',')[:, 1]

        # load sensor config
        self.cfg = OmegaConf.load(str(self.root_dir / 'sensor_specs.yaml'))
        self.cfg.speed_of_sound = 331.4 + 0.6 * self.cfg.temperature

        # replicate ground truth data to match sample number
        gt_scale = len(self.samples_env) // len(self.gt_env)
        self.gt_env = [gt for el in self.gt_env for gt in [el,]*gt_scale]
        self.gt_iq = [gt for el in self.gt_iq for gt in [el,]*gt_scale]
        self.gt_positions = [gt for el in self.gt_positions for gt in [el,]*gt_scale]
        
        assert len(self.samples_env) == len(self.gt_env) == len(self.samples_iq) == len(self.gt_iq) == len(self.gt_positions), 'inconsistent sample numbers'

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

        data_len = iq_data.shape[0]
        x = np.linspace(0, data_len/fs, num=data_len, endpoint=True)
        t = np.linspace(0, data_len/fs, num=int(data_len*rescale_factor), endpoint=True)
        
        f = interp1d(x, iq_data, axis=0)
        y = f(t)

        rf_data = y * np.exp(2j*np.pi*fc*t)

        return rf_data.real

    def __len__(self):
        return len(self.gt_positions)

    def __getitem__(self, idx):

        # load data
        envelope_data = np.loadtxt(self.samples_env[idx])
        iq_data = np.loadtxt(self.samples_iq[idx])
        envelope_gt = np.loadtxt(self.gt_env[idx])
        iq_gt = np.loadtxt(self.gt_iq[idx])
        gt_position = self.gt_positions[idx]

        # convert distance to travel time and sample index (GT position is [mm])
        toa = 2*(gt_position*1e-3) / self.cfg.speed_of_sound
        sample_position = toa * self.cfg.fhz_sample

        # convert to complex numbers
        iq_data = iq_data[:, 0] + 1j * iq_data[:, 1]
        iq_gt = iq_gt[:, 0] + 1j * iq_gt[:, 1]

        # convert to radio-frequency signal
        rf_data = self.iq2rf(iq_data, fc=self.cfg.fhz_carrier, fs=self.cfg.fhz_sample, rescale_factor=self.rf_scale_factor)
        rf_gt = self.iq2rf(iq_gt, fc=self.cfg.fhz_carrier, fs=self.cfg.fhz_sample, rescale_factor=self.rf_scale_factor)

        return envelope_data, rf_data, envelope_gt, rf_gt, sample_position

if __name__ == '__main__':

    script_path = Path(__file__).parent.resolve()
    dataset = ChirpDataset(script_path / 'stof_chirp101_dataset', 'test')

    from torch.utils.data import DataLoader
    loader_args = dict(batch_size=2, num_workers=1, pin_memory=False)
    data_loader = DataLoader(dataset, shuffle=False, **loader_args)

    for batch_idx, batch_data in enumerate(data_loader):
        print(batch_idx)
        print(batch_data)

        envelope_data, rf_data, envelope_gt, rf_gt, sample_position = batch_data

        fs = dataset.cfg.fhz_sample
        rf_scale_factor = dataset.rf_scale_factor
        data_len1 = envelope_data.shape[-1]
        data_len2 = rf_gt.shape[-1]
        x = np.linspace(0, data_len1/fs, num=data_len1, endpoint=True)
        t = np.linspace(0, data_len2/fs/rf_scale_factor, num=data_len2, endpoint=True)

        import matplotlib.pyplot as plt
        plt.plot(t, rf_data[0])
        plt.plot(t, rf_data[0], '.')
        plt.plot(t, rf_gt[0])
        plt.plot(x, envelope_data[0])
        plt.plot(x, envelope_gt[0])
        plt.plot([t[(sample_position[0]*rf_scale_factor).round().int()],]*2, [-.8*rf_data[0].max(), .8*rf_data[0].max()], linestyle='dashed')
        plt.show()