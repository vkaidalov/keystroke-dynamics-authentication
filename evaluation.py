import os
import math
import argparse
from typing import Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anomaly_detector import AbstractAnomalyDetector
from anomaly_detector import EuclideanAnomalyDetector
from anomaly_detector import NearestNeighborMahalanobisAnomalyDetector
from anomaly_detector import MahalanobisAnomalyDetector
from anomaly_detector import ManhattanAnomalyDetector
from anomaly_detector import ManhattanScaledAnomalyDetector
from anomaly_detector import OneClassSvmAnomalyDetector
from anomaly_detector import OutlierCountAnomalyDetector


anomaly_detector_classes: dict[str, Type[AbstractAnomalyDetector]] = {
    'euclidean': EuclideanAnomalyDetector,
    'nn_mahalanobis': NearestNeighborMahalanobisAnomalyDetector,
    'mahalanobis': MahalanobisAnomalyDetector,
    'manhattan': ManhattanAnomalyDetector,
    'manhattan_scaled': ManhattanScaledAnomalyDetector,
    'one_class_svm': OneClassSvmAnomalyDetector,
    'outlier_count': OutlierCountAnomalyDetector
}

parser = argparse.ArgumentParser()
parser.add_argument("--detector")
parser.add_argument("--exclude_dd_feature", type=bool)
args = parser.parse_args()

detector_name: str = args.detector or 'manhattan'
exclude_dd_feature: bool = args.exclude_dd_feature or False

print(detector_name, 'exclude_dd_feature' * exclude_dd_feature)

detector_class: Type[AbstractAnomalyDetector] = anomaly_detector_classes[detector_name]

output_dir_name = 'output'
detector_dir_name = f'{output_dir_name}/{detector_name}'

if exclude_dd_feature:
    detector_dir_name += "_no_dd"

samples = pd.read_csv('DSL-StrongPasswordData.csv')

if exclude_dd_feature:
    samples = samples.loc[:, ~samples.columns.str.startswith('DD')]

subject_ids: np.ndarray = samples['subject'].unique()
feature_columns_slice = slice(3, None)
# Take 5 first samples from each subject, sort the result to "iterate"
# over subjects.
common_neg_test_data = samples.groupby('subject').head(5)\
            .sort_values(by=['sessionIndex', 'rep'])

stats = []

for training_data_size in [
    *range(5, 51, 5),
    *range(60, 101, 10),
    *range(120, 201, 20)
]:
    equal_error_rates = np.array([])
    zero_miss_false_alarm_rates = np.array([])

    training_size_dir_name = f'{detector_dir_name}/train_{training_data_size}'
    os.makedirs(training_size_dir_name, exist_ok=True)

    for subject_id in subject_ids:
        print(subject_id, end=' ')
        curr_subject_samples = samples[samples['subject'] == subject_id]
        train_data = curr_subject_samples.iloc[:training_data_size].iloc[:, feature_columns_slice]
        pos_test_data = curr_subject_samples.iloc[training_data_size:2 * training_data_size].iloc[:, feature_columns_slice]
        # The 'sessionIndex' sort criterion can actually be skipped as we
        # take 5 samples from the first session of each user only.
        # Also, the first (training_data_size * 1.25) samples are taken to keep the ratio
        # of inliers and outliers across all tests. Sorting by ('sessionIndex', 'rep')
        # makes the outlier samples more diverse, so that as many user profiles as possible
        # are tested.
        neg_test_data = common_neg_test_data[common_neg_test_data['subject'] != subject_id]\
            .head(math.ceil(training_data_size * 1.25))\
            .iloc[:, feature_columns_slice]

        ad = detector_class()
        ad.train(train_data)
        pos_test_data_scores = ad.score(pos_test_data)
        neg_test_data_scores = ad.score(neg_test_data)

        # print(pos_test_data_scores)
        # print(neg_test_data_scores)

        # min-max normalization, so that all scores are in [0, 1]
        # max_value = max(pos_test_data_scores.max().max(), neg_test_data_scores.max().max())
        # min_value = min(pos_test_data_scores.min().min(), neg_test_data_scores.min().min())
        # pos_test_data_scores = (pos_test_data_scores - min_value) / (max_value - min_value)
        # neg_test_data_scores = (neg_test_data_scores - min_value) / (max_value - min_value)

        # print(pos_test_data_scores)
        # print(neg_test_data_scores)

        true_positive = np.array([])
        false_negative = np.array([])
        true_negative = np.array([])
        false_positive = np.array([])

        for _, row in pd.concat([pos_test_data_scores, neg_test_data_scores]).sort_values(by='score').iterrows():
            threshold = row['score']
            true_positive = np.append(true_positive, pos_test_data_scores[pos_test_data_scores['score'] <= threshold].shape[0])
            false_negative = np.append(false_negative, pos_test_data_scores[pos_test_data_scores['score'] > threshold].shape[0])
            true_negative = np.append(true_negative, neg_test_data_scores[neg_test_data_scores['score'] > threshold].shape[0])
            false_positive = np.append(false_positive, neg_test_data_scores[neg_test_data_scores['score'] <= threshold].shape[0])

        true_positive_rate = true_positive / (true_positive + false_negative)
        false_positive_rate = false_positive / (false_positive + true_negative)
        # print(true_positive)
        # print(false_negative)
        # print(true_positive_rate)
        # print(false_positive_rate)

        # sorted_indices = np.argsort(false_positive_rate)
        # true_positive_rate = true_positive_rate[sorted_indices]
        # false_positive_rate = false_positive_rate[sorted_indices]

        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate)

        equal_error_rate_idx = np.argmin(np.abs((1 - true_positive_rate) - false_positive_rate))
        equal_error_rate = ((1 - true_positive_rate[equal_error_rate_idx]) + false_positive_rate[equal_error_rate_idx]) / 2

        # Zero-miss false-alarm rate
        zmfar = np.min(false_positive_rate[true_positive_rate == 1])
        zmfar_idx = np.where(false_positive_rate == zmfar)[0][0]

        plt.plot(
            false_positive_rate[zmfar_idx],
            true_positive_rate[zmfar_idx],
            'o', color='black'
        )
        plt.plot(
            false_positive_rate[equal_error_rate_idx],
            true_positive_rate[equal_error_rate_idx],
            'o', color='black'
        )

        roc_curve_file_path = f'{training_size_dir_name}/roc_{subject_id}.png'
        plt.savefig(roc_curve_file_path)
        plt.close()

        equal_error_rates = np.append(equal_error_rates, equal_error_rate)
        zero_miss_false_alarm_rates = np.append(zero_miss_false_alarm_rates, zmfar)
        # print(equal_error_rates)
        # print(zero_miss_false_alarm_rates)
    print()

    # print(equal_error_rates)
    _, ax = plt.subplots()
    n, _, _ = ax.hist(equal_error_rates, equal_error_rates.size)
    ax.set_yticks(np.arange(0, max(n) + 1, 1))

    eer_hist_file_path = f'{training_size_dir_name}/eer_hist.png'
    plt.savefig(eer_hist_file_path)
    plt.close()

    _, ax = plt.subplots()
    n, _, _ = ax.hist(zero_miss_false_alarm_rates, zero_miss_false_alarm_rates.size)
    ax.set_yticks(np.arange(0, max(n) + 1, 1))

    zmfar_hist_file_path = f'{training_size_dir_name}/zmfar_hist.png'
    plt.savefig(zmfar_hist_file_path)
    plt.close()

    new_stats_dict = {
        'trainset_size': training_data_size,
        'eer_avg': np.mean(equal_error_rates),
        'eer_std': np.std(equal_error_rates),
        'zmfar_avg': np.mean(zero_miss_false_alarm_rates),
        'zmfar_std': np.std(zero_miss_false_alarm_rates)
    }
    print(new_stats_dict)
    stats.append(new_stats_dict)

stats = pd.DataFrame(stats)
stats.to_csv(f'{detector_dir_name}/stats.csv', index=False)
