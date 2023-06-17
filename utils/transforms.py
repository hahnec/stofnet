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

    def forward(self, waveform: (ndarray, Tensor)) -> Tensor:
        
        if isinstance(waveform, ndarray): waveform = Tensor(waveform)

        rand_num = clamp(randn(1), 0, 1)+.5 if self.normal_distribution else rand(1)

        if self.gain_type == "amplitude":
            waveform = waveform * self.gain * rand_num

        if self.gain_type == "db":
            waveform = F.gain(waveform, self.gain * rand_num)

        if self.gain_type == "power":
            waveform = F.gain(waveform, 10 * math.log10(self.gain * rand_num))
        
        return waveform #clamp(waveform, -1, 1)


class NormalizeVol(Module):
    def __init__(self):
        super(NormalizeVol, self).__init__()

    def forward(self, waveform: (ndarray, Tensor)) -> (ndarray, Tensor):

        return waveform/abs(waveform).max()
