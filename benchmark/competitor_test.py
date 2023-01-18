import sys, os, shutil
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from benchmark.test_utils import evaluate_clasp, evaluate_floss, evaluate_binseg, evaluate_window, evaluate_bocd, evaluate_candidate # ,evaluate_espresso


def evaluate_competitor(exp_path, n_jobs, verbose):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    os.mkdir(exp_path)

    competitors = [
        ("ClaSP", evaluate_clasp),
        ("FLOSS", evaluate_floss),
        # ("ESPRESSO", evaluate_espresso),
        ("BinSeg", evaluate_binseg),
        ("Window", evaluate_window),
        ("BOCD", evaluate_bocd)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("ClaSP", "FLUSS", "FLOSS"):
            columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score", "profile"]

        df = evaluate_candidate(
            candidate_name,
            eval_func=eval_func,
            columns=columns,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = 4, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor(exp_path, n_jobs, verbose)