import pandas as pd
from sklearn.svm import OneClassSVM

from anomaly_detector import AbstractAnomalyDetector


class OneClassSvmAnomalyDetector(AbstractAnomalyDetector):
    def __init__(self):
        self._is_trained: bool = False
        self.clf = OneClassSVM(nu=0.5)

    def train(self, train_data: pd.DataFrame) -> None:
        # Set gamma to the number of features
        self.clf.gamma = train_data.shape[1]
        self.clf.fit(train_data)
        self._is_trained = True

    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('detector is not trained')

        # Signed distance is positive for an inlier and negative for an outlier.
        distances = self.clf.decision_function(test_data)

        # Invert sign, so that the larger the value, the larger the distance from
        # the hyperplane.
        anomaly_scores = -distances

        return pd.DataFrame({'score': anomaly_scores})
