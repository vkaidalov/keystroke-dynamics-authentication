import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from anomaly_detector import AbstractAnomalyDetector


class MahalanobisAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._means: pd.DataFrame = pd.DataFrame()
        self._inv_cov_matrix: pd.DataFrame = pd.DataFrame()

    def train(self, train_data: pd.DataFrame) -> None:
        means_series: pd.Series = train_data.mean()
        self._means = means_series.values.flatten()
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
        for index, row in test_data.iterrows():
            scores.append(
                mahalanobis(row, self._means, self._inv_cov_matrix)
            )

        return pd.DataFrame({'score': scores})
