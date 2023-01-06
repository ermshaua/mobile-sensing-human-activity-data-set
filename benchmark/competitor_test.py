import sys, os, shutil
sys.path.insert(0, "../")

import numpy as np
np.random.seed(1379)

from benchmark.test_utils import evaluate_clasp, evaluate_floss, evaluate_binseg, evaluate_window, evaluate_candidate


def evaluate_competitor(exp_path, n_jobs, verbose):
    name = f"competitor"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    competitors = [
        ("ClaSP", evaluate_clasp),
        ("FLOSS", evaluate_floss),
        ("BinSeg", evaluate_binseg),
        ("Window", evaluate_window)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("ClaSP", "FLOSS"):
            columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score", "profile"]

        df = evaluate_candidate(
            candidate_name,
            eval_func=eval_func,
            columns=columns,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{name}/{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = 60, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor(exp_path, n_jobs, verbose)