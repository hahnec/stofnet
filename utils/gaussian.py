import numpy as np


def gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-size // 2 + 1, size // 2, size)
    kernel = np.exp(-np.power(x / sigma, 2) / 2)
    return kernel / np.sum(kernel)