import numpy as np
import pandas as pd
import daproli as dp

from src.competitors.clasp.segmentation import clasp_segmentation
from src.competitors.floss import floss, fluss
from src.competitors.binseg import binseg
from src.competitors.window import window
from src.competitors.espresso import espresso
from src.competitors.bocd import bocd
from src.utils import load_mosad_dataset
from benchmark.metrics import f_measure, covering
from tqdm import tqdm


def evaluate_clasp(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    profile, window_size, found_cps, scores = clasp_segmentation(ts, n_change_points=len(cps), offset=int(250/ts.shape[0]), **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3), profile.tolist()


def evaluate_fluss(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    profile, found_cps = fluss(ts, sample_rate, n_cps=len(cps), return_cac=True, **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3), profile.tolist()


def evaluate_floss(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    profile, found_cps = floss(ts, 20*sample_rate, sample_rate, n_cps=len(cps), return_cac=True, **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3), profile.tolist()


def evaluate_espresso(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    found_cps = espresso(ts, sample_rate, n_cps=len(cps), **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3)


def evaluate_binseg(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    found_cps = binseg(ts, n_cps=len(cps), offset=int(250/ts.shape[0]), **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3)


def evaluate_window(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    found_cps = window(ts, 5*sample_rate, n_cps=len(cps), offset=int(250/ts.shape[0]), **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3)


def evaluate_bocd(dataset, routine, subject, sensor, sample_rate, cps, activities, ts, **seg_kwargs):
    found_cps = bocd(ts, n_cps=len(cps), **seg_kwargs)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3)


def evaluate_candidate(candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    df = load_mosad_dataset()

    df_cand = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df.iterrows()), disable=verbose<1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(f"{candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}")
    return df_cand