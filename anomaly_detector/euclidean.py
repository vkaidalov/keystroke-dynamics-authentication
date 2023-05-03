from scipy.spatial.distance import euclidean
import pandas as pd

from anomaly_detector import AbstractAnomalyDetector


class EuclideanAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self._means: pd.DataFrame | None = None

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
                euclidean(self._means, row.values.flatten())
            )

        return pd.DataFrame({'score': scores})


if __name__ == "__main__":
    ead = EuclideanAnomalyDetector()
    train_data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [3, 4, 6],
        'c': [5, 6, 9]
    })
    ead.train(train_data)
    test_data = pd.DataFrame({
        'a': [2, 3],
        'b': [4.5, 1],
        'c': [6.5, 2]
    })
    scores = ead.score(test_data)
    print(scores)
