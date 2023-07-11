from torch import Tensor, rand, randn, clamp
from torch.nn import Module
from torchaudio import functional as F
from torchaudio.transforms import Vol
from numpy import ndarray, random, linspace, pad
from scipy.interpolate import interp1d


class RandomVol(Vol):
    def __init__(self, 
        gain: float = 1., 
        gain_type: str = 'amplitude',
        normal_distribution: bool = False, 
    ):
        super(RandomVol, self).__init__(gain, gain_type)

        self.normal_distribution = normal_distribution

    def forward(self, waveform: (ndarray, Tensor), *args, **kwargs) -> Tensor:
        
        if isinstance(waveform, ndarray): waveform = Tensor(waveform)

        rand_num = clamp(randn(1), 0, 1)+.5 if self.normal_distribution else rand(1)

        if self.gain_type == "amplitude":
            waveform = waveform * self.gain * rand_num

        if self.gain_type == "db":
            waveform = F.gain(waveform, self.gain * rand_num)

        if self.gain_type == "power":
            waveform = F.gain(waveform, 10 * math.log10(self.gain * rand_num))
        
        return waveform, *args, *kwargs #clamp(waveform, -1, 1)


class NormalizeVol(Module):
    def __init__(self):
        super(NormalizeVol, self).__init__()

    def forward(self, waveform: (ndarray, Tensor), *args, **kwargs) -> (ndarray, Tensor):

        norm = waveform/abs(waveform).max()

        if len(args) == 0 and len(kwargs) == 0:
            return norm
        else:
            return norm, *args, *kwargs


class AddNoise(Module):
    def __init__(self,
        snr = 40,
    ):
        super(AddNoise, self).__init__()

        self.snr = snr

    def forward(self, waveform: (ndarray, Tensor), *args, **kwargs) -> (ndarray, Tensor):
        
        mean, spread = (.5, 2) if (waveform < 0).any() else (0, 1)
        noise = spread * (random.rand(*waveform.shape) - mean)
        snr_noise = noise * (10**(-self.snr/10) * (sum(waveform**2) / sum(noise**2)))**.5

        if len(args) == 0 and len(kwargs) == 0:
            return waveform + snr_noise
        else:
            return waveform + snr_noise, *args, *kwargs


class CropChannelData(Module):
    def __init__(self,
        ratio: float = None,
        resize: bool = False,
    ):
        super(CropChannelData, self).__init__()

        self.ratio = ratio
        self.resize = resize

    @staticmethod
    def upscale_1d(data, rescale_factor):

        x = linspace(0, data.size, num=data.size, endpoint=True)
        t = linspace(0, data.size, num=int(data.size*rescale_factor), endpoint=True)
        y = interp1d(x, data, axis=0)(t)

        return y

    def forward(self, waveform: (ndarray, Tensor), gt: (float, ndarray, Tensor), *args, **kwargs) -> (ndarray, Tensor):

        self.ratio = float(rand(1)) if self.ratio is None else self.ratio

        # validate ratio
        if not (0 < self.ratio < 1):
            return waveform, gt, *args, *kwargs

        # variable init
        width = int(round(waveform.size * self.ratio))
        ref = int(round(gt))

        # define start and end indices
        start = max(0, ref-width//2)
        end = min(ref+width//2, waveform.size)
        if end == waveform.size: start = end - width
        if start == 0: end = width

        # randomized crop window movement within boundaries
        max_dist = min(ref-start, end-ref)  # keep window around reference index
        shift = random.randint(-min(start, max_dist//2), min(waveform.size-end, max_dist//2))
        start += shift
        end += shift

        # cropping
        cropped = waveform[start:end]
        gt -= start
        assert cropped.size == width

        # resize or pad for consistent waveform length (as required by some models)
        if self.resize:
            rescale_factor = waveform.size / cropped.size
            cropped = self.upscale_1d(cropped, rescale_factor)
            gt *= rescale_factor
        else:
            pad_len = waveform.size - cropped.size
            cropped = pad(cropped, (0, pad_len), mode='constant')

        assert cropped.size == waveform.size

        return cropped, gt, *args, *kwargs
