import pandas as pd
from sklearn.svm import OneClassSVM

from anomaly_detector import AbstractAnomalyDetector


class OneClassSvmAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self.clf = OneClassSVM(nu=0.5)

    def train(self, train_data: pd.DataFrame) -> None:
        self.clf.fit(train_data)
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        distances = self.clf.decision_function(test_data)
        anomaly_scores = -distances

        return pd.DataFrame({'score': anomaly_scores})
