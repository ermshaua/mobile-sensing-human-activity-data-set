import numpy as np

from bocd import BayesianOnlineChangePointDetection, ConstantHazard, StudentT


def bocd(ts, n_cps, return_profile=False):
    # needs normalizing to not run into FloatingPointError
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    bc = BayesianOnlineChangePointDetection(ConstantHazard(100), StudentT())

    profile = np.empty(ts.shape)

    for idx, timepoint in enumerate(ts):
        bc.update(timepoint)
        profile[idx] = bc.rt

    diff = np.diff(profile)

    cps = np.where(diff < 0)[0]
    scores = diff[cps]
    cps = cps[np.argsort(scores)][:n_cps]

    if return_profile is True:
        return diff, cps

    return cps





