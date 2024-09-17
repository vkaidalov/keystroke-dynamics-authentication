import argparse
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from anomaly_detector import (
    AbstractAnomalyDetector,
    ANOMALY_DETECTOR_CLASSES
)

parser = argparse.ArgumentParser()
parser.add_argument("--detector")
parser.add_argument("--exclude_dd_feature", type=bool)
args = parser.parse_args()

detector_name: str = args.detector or 'manhattan'
exclude_dd_feature: bool = args.exclude_dd_feature or False

print(detector_name, 'exclude_dd_feature' * exclude_dd_feature)

detector_class: Type[AbstractAnomalyDetector] = ANOMALY_DETECTOR_CLASSES[detector_name]

output_dir_name = 'output'
detector_dir_path = f'{output_dir_name}/{detector_name}'

for training_data_size in [
    *range(5, 51, 5),
    *range(60, 101, 10),
    *range(120, 201, 20)
]:
    training_size_dir_path = f'{detector_dir_path}/train_{training_data_size}'

    err_hist_data_file_path = f'{training_size_dir_path}/eer_hist.pkl'
    with open(err_hist_data_file_path, 'rb') as f:
        equal_error_rates = pickle.load(f)

    zmfar_hist_data_file_path = f'{training_size_dir_path}/zmfar_hist.pkl'
    with open(zmfar_hist_data_file_path, 'rb') as f:
        zero_miss_false_alarm_rates = pickle.load(f)

    _, ax = plt.subplots()
    n, _, _ = ax.hist(equal_error_rates, equal_error_rates.size)
    ax.set_yticks(np.arange(0, max(n) + 1, 1))

    eer_hist_img_file_path = f'{training_size_dir_path}/eer_hist.png'
    plt.savefig(eer_hist_img_file_path)
    plt.close()

    _, ax = plt.subplots()
    n, _, _ = ax.hist(zero_miss_false_alarm_rates, zero_miss_false_alarm_rates.size)
    ax.set_yticks(np.arange(0, max(n) + 1, 1))

    zmfar_hist_img_file_path = f'{training_size_dir_path}/zmfar_hist.png'
    plt.savefig(zmfar_hist_img_file_path)
    plt.close()
