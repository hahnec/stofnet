import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.hilbert import hilbert_transform
from utils.sample_shuffle import SampleShuffle1D


class StofNet(nn.Module):

    def __init__(self, upsample_factor, hilbert_opt=False):
        super(StofNet, self).__init__()

        if hilbert_opt:
            self.add_hilbert = lambda x: torch.cat([abs(hilbert_transform(x)), x], dim=1)
            self.conv1 = nn.Conv1d(2, 64, 9, 1, 4)
        else:
            self.add_hilbert = None
            self.conv1 = nn.Conv1d(1, 64, 9, 1, 4)

        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv8 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv9 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv10 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv11 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv12 = nn.Conv1d(64, 64, 3, 1, 1)

        self.conv13 = nn.Conv1d(64, upsample_factor, 3, 1, 1)
        
        self.sample_shuffle = SampleShuffle1D(upsample_factor)
        
    def forward(self, x):

        x = self.add_hilbert(x) if self.add_hilbert is not None else x
        x = F.relu(self.conv1(x))
        res1=x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x=torch.add(res1,x)
        res3=x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x=torch.add(res3,x)
        res5=x
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x=torch.add(res5,x)
        res7=x
        x = F.relu(self.conv8(x))
        x = self.conv9(x)
        x=torch.add(res7,x)
        res9=x
        x = F.relu(self.conv10(x))
        x = self.conv11(x)
        x=torch.add(res9,x)
        x = self.conv12(x)
        x=torch.add(res1,x)
        
        x = self.sample_shuffle(self.conv13(x))
        
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight)
        init.orthogonal(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv5.weight)
        init.orthogonal(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv7.weight)
        init.orthogonal(self.conv8.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv9.weight)
        init.orthogonal(self.conv10.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv11.weight)
        init.orthogonal(self.conv12.weight)
        init.orthogonal(self.conv13.weight)
        
        
        
        