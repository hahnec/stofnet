import numpy as np
from scipy import signal


def rf2iq(s, t, fc, fs=None): 

    iq = s * np.exp(-2j*np.pi*fc*t)
    
    if fs is not None:
        wn = min(2*fc/fs, 0.5)
        b, a = signal.butter(5, wn, btype='lowpass')
        iq = 2*signal.filtfilt(b, a, iq, axis=-1)

    return iq

iq2rf = lambda s, t, fc: s * np.exp(+2j*np.pi*fc*t)

safe_div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe')
#iq2mp = lambda s: [(s.real**2+s.imag**2)**.5, np.arctan(safe_div(s.imag, s.real))]
iq2mp = lambda s: [abs(s), np.angle(s)]

mp2rf = lambda m, p, t, fc: m * np.cos(2*np.pi*fc*t+p)
#mp2iq = lambda m, p, t, fc, fs: rf2iq(mp2rf(m, p, fc, t), t, fc, fs)
mp2iq = lambda m, p, t=None, fc=None, fs=None: m * np.exp(1j*p)


if __name__ == '__main__':

    x = np.linspace(-1e-2, 1e-2, int(2e3))
    fc = 1e3
    fs = 1/np.diff(x)[0]

    from multimodal_emg.models.wave_model import emg_wave_model

    s = emg_wave_model(
            alpha = 1,
            mu = 0,
            sigma = 2.0/fc,
            eta = 2,
            fkhz = fc/1000,
            phi = 0,
            x=x)

    iq = rf2iq(s, x, fc, fs)
    m, p = iq2mp(iq)
    rf = iq2rf(iq[::5], fc, x[::5])

    rfiqmp = iq2rf(mp2iq(m, p, x, fc, fs), x, fc)

    rfmp = mp2rf(m, p, x, fc)
    m **= 2
    m /= m.max()
    rfmp_proc = mp2rf(m, p, x, fc)

    import matplotlib.pyplot as plt
    plt.plot(x, s, color='k', label='original RF')
    plt.plot(x, abs(iq), color='m', label='abs(iq)')
    plt.plot(x[::5], rf, color='orange', linestyle='dashdot', label='RF reconstructed from IQ')
    plt.plot(x, rfmp, color='red', linestyle='dashed', label='RF reconstructed from MP')
    plt.plot(x, rfiqmp, color='blue', linestyle='dotted', label='RF reconstructed from IQMP')
    plt.plot(x, rfmp_proc, color='green', linestyle='dashed', label='RF reconstructed from MP processed')
    plt.legend(loc='upper left')
    plt.show()
