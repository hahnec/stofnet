import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.hilbert import hilbert_transform, HilbertTransform
from utils.sample_shuffle import SampleShuffle1D


class StofNet(nn.Module):

    def __init__(self, num_features=64, upsample_factor=4, fs=None, hilbert_opt=True, concat_oscil=True):
        super(StofNet, self).__init__()
    
        # input signal handling
        in_channels = 1
        self.sinc_filter = None
        self.hilb_filter = None
        if fs:
            from models.sincnet import SincConv_fast
            in_channels = 128
            kernel_size = 1024
            self.sinc_filter = SincConv_fast(in_channels, kernel_size=kernel_size+1, sample_rate=fs, in_channels=1, padding=kernel_size//2)
        elif hilbert_opt:
            if hilbert_opt and concat_oscil: in_channels = 2
            self.hilb_filter = HilbertTransform(concat_oscil=concat_oscil)

        # init first and last layer
        self.conv1 = nn.Conv1d(in_channels, num_features, 9, 1, 4)
        self.conv13 = nn.Conv1d(num_features, upsample_factor, 3, 1, 1)

        # init semi-global block
        self.semi_global_block = SemiGlobalBlock(num_features, num_features, 80)

        # init remaining layers
        for i in range(2, 13):
            setattr(self, f'conv{i}', nn.Conv1d(num_features, num_features, 7, 1, padding='same'))

        # shuffle feature channels to high resolution output
        self.sample_shuffle = SampleShuffle1D(upsample_factor)

        # indices of layers where residual connections are added and ReLU is not used
        self.residual_layers = [3, 5, 7, 9, 11, 12, 13]

    def forward(self, x):
        
        # input signal handling
        x = self.sinc_filter(x) if self.sinc_filter else x
        x = self.hilb_filter(x) if self.hilb_filter is not None else x
        
        # first layer
        x = F.relu(self.conv1(x))

        # semi-global block
        x = self.semi_global_block(x)

        # iterate through convolutional layers
        res, res1 = x, x
        for i in range(2, 12):
            # get corresponding convolutional layer
            conv = getattr(self, f'conv{i}')
            # pass data to layer considering residual connection
            x = torch.add(res, conv(x)) if i in self.residual_layers else F.leaky_relu(conv(x))
            # store residuals
            if i in self.residual_layers: res = x

        # second last layer
        x = torch.add(res1, self.conv12(x))

        # last layer and shuffling
        x = self.sample_shuffle(self.conv13(x))
        
        return x

    def _initialize_weights(self):

        return None

        for i in range(1, 14):
            if i not in self.residual_layers:
                init.orthogonal(getattr(self, f'conv{i}').weight, init.calculate_gain('relu'))
            else:
                init.orthogonal(getattr(self, f'conv{i}').weight)


class SemiGlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample_scale=2):
        super(SemiGlobalBlock, self).__init__()

        self.sample_scale = sample_scale
        self.feat_scale = max(1, sample_scale//10)

        # Contracting path
        self.contract_conv = nn.Conv1d(in_channels, self.feat_scale*out_channels, kernel_size=5, stride=1, padding=1)
        self.contract_relu = nn.LeakyReLU()
        #self.attention = AttentionBlock(self.feat_scale*out_channels, self.feat_scale*out_channels)
        self.contract_pool = nn.MaxPool1d(kernel_size=sample_scale, stride=sample_scale)

        # Expanding path
        self.expand_conv = nn.Conv1d(self.feat_scale*out_channels, out_channels, kernel_size=5, stride=1, padding=1)
        self.expand_relu = nn.LeakyReLU()
        self.expand_upsample = nn.Upsample(scale_factor=sample_scale, mode='nearest')

    def forward(self, x):
        # Contracting path
        x_scale = self.contract_conv(x)
        x_scale = self.contract_relu(x_scale)
        #x_scale = self.attention(x_scale)  # Apply attention
        x_scale = self.contract_pool(x_scale)

        # Expanding path
        x_scale = self.expand_conv(x_scale)
        x_scale = self.expand_relu(x_scale)
        x_scale = self.expand_upsample(x_scale)

        # Adjust padding for correct output size
        padding = max(0, x.size(-1) - x_scale.size(-1))
        x_scale = nn.functional.pad(x_scale, (padding // 2, padding // 2))

        # Skip connection via addition
        x = torch.add(x, x_scale)
        
        return x


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionBlock, self).__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Compute query, key, and value
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)  # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_weights = self.softmax(scores)  # (batch_size, seq_len, seq_len)

        # Apply attention weights to value
        attended_values = torch.bmm(attention_weights, v)  # (batch_size, seq_len, hidden_dim)

        return attended_values