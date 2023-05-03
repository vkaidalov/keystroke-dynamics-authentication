import pandas as pd
from scipy.spatial.distance import cityblock

from anomaly_detector import AbstractAnomalyDetector


class ManhattanAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._means: pd.DataFrame = pd.DataFrame()

    def train(self, train_data: pd.DataFrame) -> None:
        means_series: pd.Series = train_data.mean()
        self._means = means_series.values.flatten()
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        scores = []
        for index, row in test_data.iterrows():
            scores.append(
                cityblock(row, self._means)
            )

        return pd.DataFrame({'score': scores})
