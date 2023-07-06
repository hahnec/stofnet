# Network design modeled after
# https://github.com/kuleshov/audio-super-res
# https://github.com/abhishyantkhare/audio-upsampling/blob/master/resnet/upnet_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def SubPixel1D(I, r):
    b, w, r = I.size()
    X = I.permute(2, 1, 0)  # (r, w, b)
    X = X.reshape(1, r * w, b)  # (1, r*w, b)
    X = X.permute(2, 1, 0)
    return X


class Kuleshov(nn.Module):
    def __init__(self, input_length, output_length, num_layers=4):
        super(Kuleshov, self).__init__()
        self.layers = num_layers
        self.input_length = input_length
        self.output_length = output_length

        n_filters = [128, 256, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

        # Downsampling layers
        for i, (l, nf, fs) in enumerate(zip(list(range(num_layers)), n_filters, n_filtersizes)):
            if i == 0:
                setattr(self, f'down_conv{i}', nn.Conv1d(1, nf, fs, stride=2))
            else:
                setattr(self, f'down_conv{i}', nn.Conv1d(n_filters[i - 1], nf, fs, stride=2))
            setattr(self, f'down_bn{i}', nn.BatchNorm1d(nf))
            setattr(self, f'down_do{i}', nn.LeakyReLU(0.2))

        # bottleneck layer
        self.bottleneck = nn.Conv1d(n_filters[-1], n_filters[-1], n_filtersizes[-1], stride=2)
        self.bottleneck_dropout = nn.Dropout(p=0.5)
        self.bottleneck_last = nn.LeakyReLU(0.2)

        # upsampling layers
        for i, (l, nf, fs) in enumerate(reversed(list(zip(
                range(num_layers), n_filters, n_filtersizes
        )))):
            if i == 0:
                setattr(self, f'up_conv{i}', nn.Conv1d(n_filters[-1], 2 * nf, fs))
            else:
                setattr(self, f'up_conv{i}', nn.Conv1d(n_filters[-i], 2 * nf, fs))
            setattr(self, f'up_bn{i}', nn.BatchNorm1d(2*nf))
            setattr(self, f'up_do{i}', nn.Dropout(p=0.5))

        # upsample
        self.subpixel = nn.PixelShuffle(2)

        # final conv layer
        self.final_conv = nn.Conv1d(n_filters[0], 2, 9)
        self.output_fc = nn.Linear(self.fc_dimensions(n_filters, n_filtersizes), output_length)

        print(self.fc_dimensions(n_filters, n_filtersizes))

    def fc_dimensions(self, n_filters, n_filtersizes):
        def conv_dims(w, k, s, p=0):
            out = (w - k + 2 * p) / s + 1.0
            out = int(out)
            return out

        def subpixel_dims(shape, r):
            H = 1
            _, C, W = shape
            shape = [1, C / pow(r, 2), H * r, W * r]
            return shape[1:]

        shape = [1, 1, self.input_length]

        dl = []

        # down convs
        for i, (nf, fs) in enumerate(zip(n_filters, n_filtersizes)):
            _, _, w = shape
            cd = conv_dims(w=w, k=fs, s=2)
            dl.append(cd)
            shape = [1, nf, cd]

        # bottleneck with conv
        _, _, w = shape
        shape = [1, n_filters[-1], conv_dims(w=w, k=n_filtersizes[-1], s=2)]

        # upsample
        for i, (nf, fs, cd) in enumerate(reversed(list(zip(n_filters, n_filtersizes, dl)))):
            _, _, w = shape

            # up conv
            shape = [1, 2 * nf, conv_dims(w=w, k=fs, s=1)]

            # subpixel
            C, H, W = subpixel_dims(shape, 2)

            # view
            C, H, W = (1, C * H, W)

            # cat
            C, H, W = (C, H, W + cd)

            shape = [C, H, W]

        # final conv
        _, _, w = shape
        w = conv_dims(w=w, k=9, s=1)

        return w * 2

    def forward(self, x):
        x = x[:, :, :self.input_length]

        downsampling_l = [x]

        for i in range(self.layers):
            conv = getattr(self, f'down_conv{i}')
            bn = getattr(self, f'down_bn{i}')
            do = getattr(self, f'down_do{i}')
            x = F.leaky_relu(conv(x))
            x = do(bn(x))
            downsampling_l.append(x)

        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)
        x = self.bottleneck_last(x)

        for i in range(self.layers):
            conv = getattr(self, f'up_conv{i}')
            bn = getattr(self, f'up_bn{i}')
            do = getattr(self, f'up_do{i}')
            x = do(bn(conv(x)))
            x = x.unsqueeze(2)
            x = self.subpixel(x)
            x = x.view(-1, x.size()[2] * x.size()[1], x.size()[3])
            l_in = downsampling_l[len(downsampling_l) - 1 - i]
            x = torch.cat((x, l_in), -1)

        x = self.final_conv(x)
        x = SubPixel1D(x, 2)
        x = x.view(x.size()[0], x.size()[1])

        x = self.output_fc(x)
        return x.unsqueeze(1)
