from torch import Tensor, rand, randn, clamp
from torch.nn import Module
from torchaudio import functional as F
from torchaudio.transforms import Vol
from numpy import ndarray


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

        return waveform/abs(waveform).max(), *args, *kwargs


class CropChannelData(Module):
    def __init__(self,
        ratio: float = None,
        resize: bool = False,
    ):
        super(CropChannelData, self).__init__()

        self.ratio = ratio
        self.resize = resize

    def forward(self, waveform: (ndarray, Tensor), gt: (float, ndarray, Tensor), *args, **kwargs) -> (ndarray, Tensor):

        self.ratio = float(randn(1)+.5) if self.ratio is None else self.ratio
        width = int(round(waveform.size * self.ratio))
        start = int(round(gt - float(randn(1)) * width))
        cropped = waveform[start:start+width]
        gt -= start

        return cropped, gt, *args, *kwargs
