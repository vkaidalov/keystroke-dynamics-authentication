import numpy as np
import pandas as pd

from anomaly_detector import AbstractAnomalyDetector


class ManhattanScaledAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._means: pd.DataFrame = pd.DataFrame()
        # Mean Absolute Deviation (MAD)
        self._mads: pd.DataFrame = pd.DataFrame()

    def train(self, train_data: pd.DataFrame) -> None:
        means_series: pd.Series = train_data.mean()
        self._means = means_series.values.flatten()
        self._mads = np.mean(np.abs(train_data - np.median(train_data, axis=0)), axis=0)
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        scores = []
        for _, row in test_data.iterrows():
            scores.append(
                np.sum(np.abs(row - self._means) / self._mads)
            )

        return pd.DataFrame({'score': scores})
