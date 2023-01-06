import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd


def load_mosad_dataset():
    desc_filename = ABS_PATH + "/../datasets/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    ts_filename = ABS_PATH + "/../datasets/data.npz"
    T = np.load(file=ts_filename)

    df = []

    for row in desc_file:
        (ts_name, sample_rate), change_points = row[:2], row[2:]
        routine, subject, sensor = ts_name.split("_")
        ts = T[ts_name]

        df.append((ts_name, int(routine[-1]), int(subject[-1]), sensor, int(sample_rate), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "routine", "subject", "sensor", "sample_rate", "change_points", "time_series"])
