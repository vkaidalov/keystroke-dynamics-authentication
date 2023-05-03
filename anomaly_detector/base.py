import abc

import pandas as pd


class AbstractAnomalyDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        """
        Train the detector using the dataset of "normal" samples.
        As soon as training is completed, the detector can be used for
        detection of anomalies in new samples.
        :param train_data: Each row is a vector of feature values.
        """
        pass

    @abc.abstractmethod
    def score(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate anomaly scores for each sample of the provided dataset.
        The detector must be already trained before calling this method.
        :param test_data: Each row must have the same number of features as
        the test dataset's rows.
        :return: A vector of scores for each given sample. Each score value is
        implementation-specific and a threshold value must be chosen respectively.
        """
        pass
