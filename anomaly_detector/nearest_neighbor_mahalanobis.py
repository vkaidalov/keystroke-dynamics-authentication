import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from anomaly_detector import AbstractAnomalyDetector


class NearestNeighborMahalanobisAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._train_data: pd.DataFrame = pd.DataFrame()
        self._inv_cov_matrix: pd.DataFrame = pd.DataFrame()

    def train(self, train_data: pd.DataFrame) -> None:
        self._train_data = train_data.copy()
        cov_matrix = train_data.cov()
        self._inv_cov_matrix = pd.DataFrame(
            # generalized inverse
            np.linalg.pinv(cov_matrix.values),
            columns=cov_matrix.columns,
            index=cov_matrix.index
        )
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        scores = []

        for _, test_row in test_data.iterrows():
            min_distance = float('inf')
            for _, train_row in self._train_data.iterrows():
                curr_distance = mahalanobis(test_row, train_row, self._inv_cov_matrix)
                min_distance = min(min_distance, curr_distance)
            scores.append(min_distance)

        return pd.DataFrame({'score': scores})
