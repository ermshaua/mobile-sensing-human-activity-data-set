import numpy as np
import pandas as pd

from src.competitors.clasp.knn import IntervalKneighbours
from numba import njit


@njit(fastmath=True, cache=True)
def binary_f1_score(y_true, y_pred):
    f1_scores = np.zeros(shape=2, dtype=np.float64)

    for label in (0,1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        f1 = 2 * (pr * re) / (pr + re)

        f1_scores[label] = f1

    return np.mean(f1_scores)


@njit(fastmath=True, cache=True)
def binary_roc_auc_score(y_score, y_true):
    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate((
        distinct_value_indices,
        np.array([y_true.size - 1])
    ))

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0): direction = -1
        else: return np.nan

    area = direction * np.trapz(tpr, fpr)
    return area


@njit(fastmath=True, cache=True)
def _labels(knn_mask, split_idx, window_size):
    n_timepoints, k_neighbours = knn_mask.shape

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = 1

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _profile(window_size, knn, score, offset):
    n_timepoints, _ = knn.shape
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)
    offset = max(10*window_size, offset)

    for split_idx in range(offset, n_timepoints - offset):
        y_true, y_pred = _labels(knn, split_idx, window_size)

        try:
            _score = score(y_true, y_pred)

            if not np.isnan(_score):
                profile[split_idx] = _score
        except:
            pass

    return profile


class ClaSPEnsemble:

    def __init__(self, window_size, n_iter=30, k_neighbours=3, score=binary_roc_auc_score, min_seg_size=None, interval_knn=None, interval=None, offset=.05, random_state=1379):
        self.window_size = window_size
        self.n_iter = n_iter
        self.k_neighbours = k_neighbours
        self.score = score
        self.min_seg_size = min_seg_size
        self.interval_knn = interval_knn
        self.interval = interval
        self.offset = offset
        self.random_state = random_state

    def fit(self, time_series):
        return self

    def _calculate_tcs(self, time_series):
        tcs = [(0, time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_iter and time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(time_series.shape[0], 2, replace=True)

            if time_series.shape[0] - lbound < area:
                area = time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(tcs, dtype=np.int64)

    def _ensemble_profiles(self, time_series):
        self.tcs = self._calculate_tcs(time_series)
        _, self.knn = self.interval_knn.knn(self.interval, self.tcs)

        n_timepoints = self.knn.shape[0]
        offset = np.int64(n_timepoints * self.offset)

        profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)  #
        bounds = np.full(shape=(n_timepoints, 3), fill_value=-1, dtype=np.int64)

        for idx, (lbound, ubound) in enumerate(self.tcs):
            tc_knn = self.knn[lbound:ubound, idx * self.k_neighbours:(idx + 1) * self.k_neighbours] - lbound

            tc_profile = _profile(self.window_size, tc_knn, self.score, offset)
            not_ninf = np.logical_not(tc_profile == -np.inf)

            tc = (ubound - lbound) / self.knn.shape[0]
            tc_profile[not_ninf] = (2 * tc_profile[not_ninf] + tc) / 3

            change_idx = profile[lbound:ubound] < tc_profile
            change_mask = np.logical_and(change_idx, not_ninf)

            profile[lbound:ubound][change_mask] = tc_profile[change_mask]
            bounds[lbound:ubound][change_mask] = np.array([idx, lbound, ubound])

        return profile, bounds

    def transform(self, time_series, interpolate=False):
        if self.min_seg_size is None:
            self.min_seg_size = int(max(10 * self.window_size, self.offset * time_series.shape[0]))

        if self.interval_knn is None:
            self.interval_knn = IntervalKneighbours(time_series, self.window_size, self.k_neighbours)

        if self.interval is None:
            self.interval = (0, time_series.shape[0])

        profile, self.bounds = self._ensemble_profiles(time_series)

        if interpolate:
            profile[np.isinf(profile)] = np.nan
            profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        return profile

    def fit_transform(self, time_series, interpolate=False):
        return self.fit(time_series).transform(time_series, interpolate=interpolate)

    def applc_tc(self, profile, change_point):
        idx, lbound, ubound = self.bounds[change_point]

        if lbound == -1:
            return None, None, None

        tc_profile = profile[lbound:ubound]
        tc_knn = self.knn[lbound:ubound, idx * self.k_neighbours:(idx + 1) * self.k_neighbours] - lbound
        tc_change_point = change_point - lbound

        return tc_profile, tc_knn, tc_change_point