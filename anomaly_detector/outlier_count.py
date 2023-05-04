import numpy as np
import pandas as pd

from anomaly_detector import AbstractAnomalyDetector


class OutlierCountAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._means: pd.DataFrame = pd.DataFrame()
        # standard deviations
        self._stds: pd.DataFrame = pd.DataFrame()

    def train(self, train_data: pd.DataFrame) -> None:
        means_series: pd.Series = train_data.mean()
        self._means = means_series.values.flatten()
        self._stds = train_data.std()
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        scores = []

        for _, row in test_data.iterrows():
            z_scores = np.abs(row - self._means) / self._stds
            scores.append(
                np.sum(z_scores > 1.96)
            )

        return pd.DataFrame({'score': scores})
