import numpy as np

import stumpy


def floss(ts, window_size, n_cps, return_cac=False):
    mp = stumpy.stump(ts, m=window_size)

    cac, cps = stumpy.fluss(mp[:, 1], L=window_size, n_regimes=n_cps+1)

    if return_cac is True:
        return cac, cps

    return cps



