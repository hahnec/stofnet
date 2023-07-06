# https://github.com/abhishyantkhare/audio-upsampling/blob/master/resnet/upnet_torch.py

import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_SAMPLE_RATE = 8000
OUTPUT_SAMPLE_RATE = 44100
SAMPLE_LENGTH = 0.5
BATCH_SIZE = 8
NUM_WORKERS = 4

INPUT_LEN = int(INPUT_SAMPLE_RATE * SAMPLE_LENGTH)
OUTPUT_LEN = int(OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH)


DTYPE_RANGES = {
    np.dtype('float32'): (-1.0, 1.0), np.dtype('int32'): (-2147483648, 2147483647),
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.dtype('float32'), 32: np.dtype('int32'), 16: np.dtype('int16'), 8: np.dtype('uint8')
}

# Network design modeled after
# https://github.com/kuleshov/audio-super-res


def SubPixel1D(I, r):
    b, w, r = I.size()
    X = I.permute(2, 1, 0)  # (r, w, b)
    X = X.reshape(1, r * w, b)  # (1, r*w, b)
    X = X.permute(2, 1, 0)
    return X


class Kuleshov(nn.Module):
    def __init__(self, input_length, output_length, num_layers=4,
                 batch_size=128, learning_rate=1e-4, b1=0.99, b2=0.999):
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
            setattr(self, f'down_do{i}', nn.Dropout(p=0.1))

        # bottleneck layer
        self.bottleneck = nn.Conv1d(n_filters[-1], n_filters[-1], n_filtersizes[-1], stride=2)
        self.bottleneck_dropout = nn.Dropout(p=0.5)
        self.bottleneck_bn = nn.BatchNorm1d(n_filters[-1])
        # x = LeakyReLU(0.2)(x)

        # upsampling layers
        for i, (l, nf, fs) in enumerate(reversed(list(zip(
                range(num_layers), n_filters, n_filtersizes
        )))):
            if i == 0:
                setattr(self, f'up_conv{i}', nn.Conv1d(n_filters[-1], 2 * nf, fs))
            else:
                setattr(self, f'up_conv{i}', nn.Conv1d(n_filters[-i], 2 * nf, fs))
            setattr(self, f'up_bn{i}', nn.BatchNorm1d(2*nf))
            setattr(self, f'up_do{i}', nn.Dropout(p=0.1))
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
        x = self.bottleneck_bn(x)

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


def upsample(model, file):
    fs, audio = wavfile.read(file)
    end_lim = int((len(audio) // INPUT_LEN) * INPUT_LEN)
    audio = audio[:end_lim]
    upsampled_audio = np.asarray([])
    print("Beginning Upsampling")
    for i in range(0, len(audio), INPUT_LEN):
        model.eval()
        torch.no_grad()
        input_chunk = audio[i:i+INPUT_LEN]
        input_chunk = torch.from_numpy(input_chunk)
        input_chunk = input_chunk.float()#.to(device)
        input_chunk = input_chunk.unsqueeze(0)
        input_chunk = input_chunk.unsqueeze(1)
        output_chunk = model.forward(input_chunk)
        output_chunk = output_chunk.view(OUTPUT_LEN).detach().numpy()
        upsampled_audio = np.append(upsampled_audio, output_chunk)
        print("Upsampled chunk {} out of {}".format(i // INPUT_LEN, end_lim // INPUT_LEN))
    print(upsampled_audio.min(), upsampled_audio.max())
    upsampled_audio = upsampled_audio.astype(np.uint8)
    wavfile.write('upsampled_' + file, OUTPUT_SAMPLE_RATE, upsampled_audio)
