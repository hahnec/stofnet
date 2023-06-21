import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted


class ChirpDataset(Dataset):
    def __init__(self, root_dir, split_dirname='test'):
        self.root_dir = Path(root_dir)
        self.split_dirname = split_dirname

        # load sample paths
        self.samples_env, self.samples_iq = self._get_file_paths(str(self.root_dir / self.split_dirname))

        # load ground truth paths
        self.gt_env, self.gt_iq = self._get_file_paths(str(self.root_dir / 'truth'))

        # load ground truth positions
        self.gt_positions = np.genfromtxt(str(self.root_dir / 'truth' / 'true_measurement_positions.csv'), delimiter=',')[:, 1]
        
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

    def __len__(self):
        return len(self.gt_positions)

    def __getitem__(self, idx):

        envelope_data = np.loadtxt(self.samples_env[idx])
        iq_data = np.loadtxt(self.samples_iq[idx])

        envelope_gt = np.loadtxt(self.gt_env[idx])
        iq_gt = np.loadtxt(self.gt_iq[idx])

        gt_position = self.gt_positions[idx]

        return envelope_data, iq_data, envelope_gt, iq_gt, gt_position

if __name__ == '__main__':

    script_path = Path(__file__).parent.resolve()
    dataset = ChirpDataset(script_path / 'stof_chirp101_dataset', 'train')

    from torch.utils.data import DataLoader
    loader_args = dict(batch_size=2, num_workers=1, pin_memory=False)
    data_loader = DataLoader(dataset, shuffle=False, **loader_args)

    for batch_idx, batch_data in enumerate(data_loader):
        print(batch_idx)
        print(batch_data)