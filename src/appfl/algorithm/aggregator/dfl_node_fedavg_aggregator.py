import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, List


class DFLNodeFedAvgAggregator(BaseAggregator):
    def __init__(
        self, model: torch.nn.Module, aggregator_config: DictConfig, logger: Any
    ):
        self.model = model
        self.logger = logger
        self.aggregator_config = aggregator_config

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(
        self,
        local_results: Union[
            Dict, OrderedDict
        ],  # res_list <- [dict(local params), dict(round info. (round no., pre val loss/acc, post val loss/acc))]
        neighbor_results: Union[
            Dict[Union[str, int], Union[Dict, OrderedDict]],
            List[Union[Dict, OrderedDict]],
        ],  # [res_list_of_neighbor1, res_list_of_neighbor2, ...]
        **kwargs,
    ):
        updated_model = OrderedDict()
        for name in self.model.state_dict().keys():
            param_agg = local_results[0][name].detach()  # local params
            for param, info in neighbor_results:  # accumulate neighbor's params
                param_agg += param[name].detach()
            updated_model[name] = torch.div(
                param_agg, len(neighbor_results) + 1
            )  # + 1 <- itself
        self.model.load_state_dict(updated_model)
        return updated_model
