import numpy as np
import stumpy

from stumpy.floss import _rea


def fluss(ts, window_size, n_cps, return_cac=False):
    mp = stumpy.stump(ts, m=window_size)

    cac, cps = stumpy.fluss(mp[:, 1], L=window_size, n_regimes=n_cps+1)

    if return_cac is True:
        return cac, cps

    return cps


def floss(ts, sliding_window_size, window_size, n_cps, return_cac=False):
    mp = stumpy.stump(ts[:sliding_window_size], m=window_size)

    stream = stumpy.floss(
        mp,
        ts[:sliding_window_size],
        m=window_size,
        L=window_size
    )

    cac = np.full(ts.shape[0], fill_value=np.inf, dtype=np.float64)

    for dx, timepoint in zip(range(sliding_window_size, ts.shape[0]), ts[sliding_window_size:]):
        stream.update(timepoint)
        window_cac = stream.cac_1d_

        cac[max(0, dx - window_cac.shape[0]+1):dx+1] = np.min([
            cac[max(0, dx - window_cac.shape[0]+1):dx+1],
            window_cac[max(0, window_cac.shape[0]-dx-1):]
        ], axis=0)

    cps = _rea(cac, L=window_size, n_regimes=n_cps+1)

    if return_cac is True:
        return cac, cps

    return cps



