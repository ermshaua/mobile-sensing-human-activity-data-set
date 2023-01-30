import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd


def load_mosad_dataset():
    cp_filename = ABS_PATH + "/../datasets/change_points.txt"
    cp_file = []

    with open(cp_filename, 'r') as file:
        for line in file.readlines(): cp_file.append(line.split(","))
        
    activity_filename = ABS_PATH + "/../datasets/activities.txt"
    activities = dict()

    with open(activity_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")
            routine, motions = line[0], line[1:]
            activities[routine] = [motion.replace("\n", "") for motion in motions]

    ts_filename = ABS_PATH + "/../datasets/data.npz"
    T = np.load(file=ts_filename)

    df = []

    for row in cp_file:
        (ts_name, sample_rate), change_points = row[:2], row[2:]
        routine, subject, sensor = ts_name.split("_")
        ts = T[ts_name]

        df.append((ts_name, int(routine[-1]), int(subject[-1]), sensor, int(sample_rate), np.array([int(_) for _ in change_points]), np.array(activities[routine[-1]]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "routine", "subject", "sensor", "sample_rate", "change_points", "activities", "time_series"])
