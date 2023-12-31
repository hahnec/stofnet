# beamforming with data explained at https://github.com/CorazzaAlexandre/PALA_Beamforming

import numpy as np


def bf_das(rf_iq, param, compound_opt=True):

    [x, z] = np.meshgrid(param['param_x'], param['param_z'])

    iq_frame = np.zeros_like(x, dtype=rf_iq.dtype)

    if not compound_opt:
        iq_frame = np.repeat(iq_frame[None, ...], 3, axis=0)

    # iterate over number of transmitted angles
    for k in range(len(param.angles_list)):
        param.theta = param.angles_list[k]
        # rf_iq dimensions: angles x samples x channels
        rf_angle = rf_iq[k, ...]
        if compound_opt:
            # accumulate receiver delay and sum
            iq_frame += bf_das_rx(rf_angle, param, x, z)
        else:
            # attach receiver delay and sum
            iq_frame[k, ...] = bf_das_rx(rf_angle, param, x, z)

    # rescale and remove imaginary components
    bmode = 20*np.log10(abs(iq_frame), where=abs(iq_frame)>0)

    # remove infs and NaNs
    bmode[np.isnan(bmode) | np.isinf(bmode)] = np.min(bmode[np.isfinite(bmode)])

    # normalize
    bmode -= bmode.max()

    return bmode


def bf_das_rx(sig, param, x, z, fnumber=1.9):

    agg_sig = np.zeros([1, x.size], dtype=sig.dtype)

    # emit delay
    # TXdelay = (1/param.c)*tan(param.theta)*abs(param.xe - param.xe(1));

    # virtual source (non-planar wave assumption)
    beta = 1e-8
    width = param.xe[-1]-param.xe[0]    # extent of the phased-array
    vsource = [-width*np.cos(param.theta) * np.sin(param.theta)/beta, -width*np.cos(param.theta)**2/beta]

    # iterate over channels
    for k in range(param.Nelements):
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:); %convention FieldII
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:) + mean(TXdelay)*param.c; %convention FieldII
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:) + mean(TXdelay-min(TXdelay))*param.c; %convention Verasonics

        # find transmit travel distances considering virtual source
        dtx = np.hypot(x.T.flatten()-vsource[0], z.T.flatten()-vsource[1]) - np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])

        # find receive travel distances
        drx = np.hypot(x.T.flatten()-param.xe[k], z.T.flatten())

        # convert overall travel distances to delay times
        tau = (dtx+drx) / param.c

        # convert phase-shift delays into sample indices (deducting blind zone?)
        idxt = (tau-param.t0) * param.fs #+ 1
        I = ((idxt<1) + (idxt>sig.shape[0]-1)).astype(bool)
        idxt[I] = 1 # arbitrary index, will be soon rejected

        idx = idxt       # floating number representing sample index for subsequent interpolation
        idxf = np.floor(idx).astype('int64') # rounded number of samples
        IDX = idx #np.repmat(idx, [1 1 sig.shape[2]]) #3e dimension de SIG: angles

        # resample at delayed sample positions (using linear interpolation)
        #f = interp1d(np.arange(sig.shape[0]), sig[..., k])
        #temp = f(idxt)
        #temp = sig[idxf, k].flatten()
        temp = sig[idxf, k].T.flatten() * (idxf+1-IDX) + sig[idxf+1, k].T.flatten() * (IDX-idxf)
        #temp = sig[idxf-1, k].T.flatten() * (idxf+1-IDX) + sig[idxf, k].T.flatten() * (IDX-idxf)

        # mask values outside index range
        temp[I] = 0

        # IQ to RF conversion
        if np.any(~np.isreal(temp)):
            temp = temp * np.exp(2*1j*np.pi*param.f0*tau)

        # F-number mask
        mask_Fnumber = abs(x-param.xe[k]) < z / fnumber/2

        # sum delayed channel signal
        agg_sig += temp.T.flatten() * mask_Fnumber.T.flatten()

    output = agg_sig.reshape(x.shape, order='F')

    return output
