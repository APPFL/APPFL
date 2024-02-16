import abc
from typing import Dict, Union, OrderedDict

class BaseAggregator:

    @abc.abstractmethod
    def aggregate(self, local_models: Union[Union[Dict, OrderedDict], Dict[Union[str, int], Union[Dict, OrderedDict]]]) -> Union[Dict, OrderedDict]:
        """
        Aggregate local model(s) from clients and return the global model
        """
        pass

    @abc.abstractmethod
    def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict]:
        """Return global model parameters"""
        pass