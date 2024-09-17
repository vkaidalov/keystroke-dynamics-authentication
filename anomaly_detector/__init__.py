from typing import Type

from anomaly_detector.base import AbstractAnomalyDetector
from anomaly_detector.euclidean import EuclideanAnomalyDetector
from anomaly_detector.nearest_neighbor_mahalanobis import NearestNeighborMahalanobisAnomalyDetector
from anomaly_detector.mahalanobis import MahalanobisAnomalyDetector
from anomaly_detector.manhattan import ManhattanAnomalyDetector
from anomaly_detector.manhattan_scaled import ManhattanScaledAnomalyDetector
from anomaly_detector.one_class_svm import OneClassSvmAnomalyDetector
from anomaly_detector.outlier_count import OutlierCountAnomalyDetector

ANOMALY_DETECTOR_CLASSES: dict[str, Type[AbstractAnomalyDetector]] = {
    'euclidean': EuclideanAnomalyDetector,
    'nn_mahalanobis': NearestNeighborMahalanobisAnomalyDetector,
    'mahalanobis': MahalanobisAnomalyDetector,
    'manhattan': ManhattanAnomalyDetector,
    'manhattan_scaled': ManhattanScaledAnomalyDetector,
    'one_class_svm': OneClassSvmAnomalyDetector,
    'outlier_count': OutlierCountAnomalyDetector
}
