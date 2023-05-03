import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anomaly_detector import EuclideanAnomalyDetector
from anomaly_detector import NearestNeighborMahalanobisAnomalyDetector
from anomaly_detector import MahalanobisAnomalyDetector
from anomaly_detector import ManhattanAnomalyDetector
from anomaly_detector import ManhattanScaledAnomalyDetector
from anomaly_detector import OneClassSvmAnomalyDetector


samples = pd.read_csv('DSL-StrongPasswordData.csv')

subject_ids: np.ndarray = samples['subject'].unique()
feature_columns_slice = slice(3, None)
equal_error_rates = np.array([])

for subject_id in subject_ids:
    curr_subject_samples = samples[samples['subject'] == subject_id]
    train_data = curr_subject_samples.iloc[:200].iloc[:, feature_columns_slice]
    pos_test_data = curr_subject_samples.iloc[200:].iloc[:, feature_columns_slice]
    neg_test_data = samples[samples['subject'] != subject_id].groupby('subject')\
        .head(5).iloc[:, feature_columns_slice]

    ad = OneClassSvmAnomalyDetector()
    ad.train(train_data)
    pos_test_data_scores = ad.score(pos_test_data)
    neg_test_data_scores = ad.score(neg_test_data)

    # min-max normalization, so that all scores are in [0, 1]
    max_value = max(pos_test_data_scores.max().max(), neg_test_data_scores.max().max())
    min_value = min(pos_test_data_scores.min().min(), neg_test_data_scores.min().min())
    pos_test_data_scores = (pos_test_data_scores - min_value) / (max_value - min_value)
    neg_test_data_scores = (neg_test_data_scores - min_value) / (max_value - min_value)

    true_positive = np.array([])
    false_negative = np.array([])
    true_negative = np.array([])
    false_positive = np.array([])

    for i in range(100):
        threshold = (i + 1) / 100
        true_positive = np.append(true_positive, pos_test_data_scores[pos_test_data_scores['score'] <= threshold].shape[0])
        false_negative = np.append(false_negative, pos_test_data_scores[pos_test_data_scores['score'] > threshold].shape[0])
        true_negative = np.append(true_negative, neg_test_data_scores[neg_test_data_scores['score'] > threshold].shape[0])
        false_positive = np.append(false_positive, neg_test_data_scores[neg_test_data_scores['score'] <= threshold].shape[0])

    true_positive_rate = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)

    sorted_indices = np.argsort(false_positive_rate)
    true_positive_rate = true_positive_rate[sorted_indices]
    false_positive_rate = false_positive_rate[sorted_indices]

    # plt.figure()
    # plt.plot(false_positive_rate, true_positive_rate)

    equal_error_rate_idx = np.argmin(np.abs((1 - true_positive_rate) - false_positive_rate))
    equal_error_rate = ((1 - true_positive_rate[equal_error_rate_idx]) + false_positive_rate[equal_error_rate_idx]) / 2

    # plt.plot(
    #     false_positive_rate[equal_error_rate_idx],
    #     true_positive_rate[equal_error_rate_idx],
    #     'o', color='black'
    # )
    #
    # plt.show()

    equal_error_rates = np.append(equal_error_rates, equal_error_rate)
    print(equal_error_rates)
    # break


print(equal_error_rates)
plt.figure()
plt.hist(equal_error_rates, equal_error_rates.size)
plt.show()
print(np.std(equal_error_rates))
print(np.mean(equal_error_rates))
