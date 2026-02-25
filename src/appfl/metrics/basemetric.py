from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def collect(self, pred, true):
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        raise NotImplementedError
