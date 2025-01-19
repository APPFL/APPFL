import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class FedAvgAggregator(BaseAggregator):
    """
    :param `model`: An optional instance of the model to be trained in the federated learning setup.
        This can be useful for aggregating parameters that does requires gradient, such as the batch
        normalization layers. If not provided, the aggregator will only aggregate the parameters
        sent by the clients.
    :param `aggregator_configs`: Configuration for the aggregator. It should be specified in the YAML
        configuration file under `aggregator_kwargs`.
    :param `logger`: An optional instance of the logger to be used for logging.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )

        if self.model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        self.step = {}

    def get_parameters(self, **kwargs) -> Dict:
        """
        The aggregator can deal with three general aggregation cases:

        - The model is provided to the aggregator and it has the same state as the global state
        [**Note**: By global state, it means the state of the model that is used for aggregation]:
            In this case, the aggregator will always return the global state of the model.
        - The model is provided to the aggregator, but it has a different global state (e.g., part of the model is shared for aggregation):
            In this case, the aggregator will return the whole state of the model at the beginning (i.e., when it does not have the global state),
            and return the global state afterward.
        - The model is not provided to the aggregator:
            In this case, the aggregator will raise an error when it does not have the global state (i.e., at the beginning), and return the global state afterward.
        """
        if self.global_state is None:
            if self.model is not None:
                return copy.deepcopy(self.model.state_dict())
            else:
                raise ValueError("Model is not provided to the aggregator.")
        return {k: v.clone() for k, v in self.global_state.items()}

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        if self.global_state is None:
            if self.model is not None:
                try:
                    self.global_state = {
                        name: self.model.state_dict()[name]
                        for name in list(local_models.values())[0]
                    }
                except:  # noqa E722
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in list(local_models.values())[0].items()
                    }
            else:
                self.global_state = {
                    name: tensor.detach().clone()
                    for name, tensor in list(local_models.values())[0].items()
                }

        self.compute_steps(local_models)

        for name in self.global_state:
            if name in self.step:
                self.global_state[name] = self.global_state[name] + self.step[name]
            else:
                param_sum = torch.zeros_like(self.global_state[name])
                for _, model in local_models.items():
                    param_sum += model[name]
                # make sure global state have the same type as the local model
                self.global_state[name] = torch.div(param_sum, len(local_models)).type(
                    param_sum.dtype
                )

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)
        return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        for name in self.global_state:
            # Skip integer parameters by averaging them later in the `aggregate` method
            if (
                self.named_parameters is not None and name not in self.named_parameters
            ) or (
                self.global_state[name].dtype == torch.int64
                or self.global_state[name].dtype == torch.int32
            ):
                continue
            self.step[name] = torch.zeros_like(self.global_state[name])

        for client_id, model in local_models.items():
            if (
                self.client_weights_mode == "sample_size"
                and hasattr(self, "client_sample_size")
                and client_id in self.client_sample_size
            ):
                weight = self.client_sample_size[client_id] / sum(
                    self.client_sample_size.values()
                )
            else:
                weight = 1.0 / len(local_models)

            for name in model:
                if name in self.step:
                    self.step[name] += weight * (model[name] - self.global_state[name])
