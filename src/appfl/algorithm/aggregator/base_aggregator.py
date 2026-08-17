import abc
from collections import OrderedDict


class BaseAggregator:
    def set_client_sample_size(self, client_id: str | int, sample_size: int):
        """Set the sample size of a client"""
        if not hasattr(self, "client_sample_size"):
            self.client_sample_size = {}
        self.client_sample_size[client_id] = sample_size

    @abc.abstractmethod
    def aggregate(
        self, *args, **kwargs
    ) -> dict | OrderedDict | tuple[dict | OrderedDict, dict]:
        """
        Aggregate local model(s) from clients and return the global model
        """

    @abc.abstractmethod
    def get_parameters(
        self, **kwargs
    ) -> dict | OrderedDict | tuple[dict | OrderedDict, dict]:
        """Return global model parameters"""
