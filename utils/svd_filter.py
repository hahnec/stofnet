import numpy as np

def svd_filter(iq, lo_cut=4, hi_cut=-1):

    init_shape = iq.shape

    # reshape into Casorati matrix
    x = np.reshape(iq, (-1, iq.shape[-1]), order='F')

    # autocorrelation matrix
    a = np.dot(x.conj().T, x)

    # calculate svd of the autocorrelated Matrix
    u, _, _ = np.linalg.svd(a)

    # calculate the singular vectors
    v = np.dot(x, u)

    # singular value decomposition
    n = np.dot(v[:, lo_cut:hi_cut], u[:, lo_cut:hi_cut].conj().T)

    # reconstruction of the final filtered matrix
    iqf = np.reshape(n, init_shape, order='F')   

    return iqf