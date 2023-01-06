import sys, time
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from src.utils import load_mosad_dataset
from benchmark.metrics import f_measure, covering
from src.visualizer import plot_profile_with_ts
from src.clasp.segmentation import segmentation

if __name__ == '__main__':
    np.seterr(all='raise')

    df = load_mosad_dataset().iloc[0:1,:]

    for _, (dataset, routine, subject, sensor, sample_rate, change_points, time_series) in df.iterrows():
        runtime = time.process_time()
        profile, window_size, found_cps, found_scores = segmentation(time_series, n_change_points=len(change_points), offset=int(200 / time_series.shape[0]))
        # profile, found_cps = floss(time_series, sample_rate, len(change_points), return_cac=True)

        runtime = time.process_time() - runtime

        f1_score = f_measure({0: change_points}, found_cps, margin=int(time_series.shape[0] * .01))  #
        covering_score = covering({0: change_points}, found_cps, time_series.shape[0])  #

        plot_profile_with_ts(dataset, time_series, profile, change_points, found_cps, show=False, save_path="../tmp/simple_test.pdf")
        print(f"{dataset}: Window Size: {sample_rate} True Change Points: {change_points}, Found Change Points: {found_cps}, F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)} Runtime: {np.round(runtime, 3)}")