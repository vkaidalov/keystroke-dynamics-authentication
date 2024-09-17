import argparse
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

if exclude_dd_feature:
    detector_dir_path += "_no_dd"

samples = pd.read_csv('DSL-StrongPasswordData.csv')

if exclude_dd_feature:
    samples = samples.loc[:, ~samples.columns.str.startswith('DD')]

subject_ids: np.ndarray = samples['subject'].unique()

for training_data_size in [
    *range(5, 51, 5),
    *range(60, 101, 10),
    *range(120, 201, 20)
]:
    training_size_dir_path = f'{detector_dir_path}/train_{training_data_size}'

    for subject_id in subject_ids:
        roc_curve_data_file_path = f'{training_size_dir_path}/roc_{subject_id}.pkl'
        with open(roc_curve_data_file_path, 'rb') as f:
            (
                true_positive_rate,
                false_positive_rate,
                zmfar_idx,
                equal_error_rate_idx
            ) = pickle.load(f)

        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.plot(
            false_positive_rate[zmfar_idx],
            true_positive_rate[zmfar_idx],
            'o', color='black'
        )
        plt.text(
            false_positive_rate[zmfar_idx],
            true_positive_rate[zmfar_idx] - 0.05,
            'Zero-Miss False Alarm Rate',
            fontsize=9
        )
        plt.plot(
            false_positive_rate[equal_error_rate_idx],
            true_positive_rate[equal_error_rate_idx],
            'o', color='black'
        )
        plt.text(
            false_positive_rate[equal_error_rate_idx],
            true_positive_rate[equal_error_rate_idx] - 0.05,
            'Equal Error Rate',
            fontsize=9
        )

        roc_curve_file_path = f'{training_size_dir_path}/roc_{subject_id}.png'
        plt.savefig(roc_curve_file_path)
        plt.close()
