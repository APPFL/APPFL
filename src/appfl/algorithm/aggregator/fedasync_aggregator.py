import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class FedAsyncAggregator(BaseAggregator):
    """
    FedAsync Aggregator class for Federated Learning.
    For more details, check paper: https://arxiv.org/pdf/1903.03934.pdf
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

        if model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name=self.aggregator_configs.get("staleness_fn", "constant"),
            **self.aggregator_configs.get("staleness_fn_kwargs", {}),
        )
        self.alpha = self.aggregator_configs.get("alpha", 0.9)
        self.global_step = 0
        self.client_step = {}
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
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Dict:
        if self.global_state is None:
            if self.model is not None:
                try:
                    self.global_state = {
                        name: self.model.state_dict()[name] for name in local_model
                    }
                except:  # noqa E722
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in local_model.items()
                    }
            else:
                self.global_state = {
                    name: tensor.detach().clone()
                    for name, tensor in local_model.items()
                }

        self.compute_steps(client_id, local_model)

        for name in self.global_state:
            if name in self.step:
                self.global_state[name] += self.step[name]
            else:
                self.global_state[name] = local_model[name]

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        self.global_step += 1
        self.client_step[client_id] = self.global_step
        return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
    ):
        """
        Compute changes to the global model after the aggregation.
        """
        if client_id not in self.client_step:
            self.client_step[client_id] = 0
        gradient_based = self.aggregator_configs.get("gradient_based", False)
        if (
            self.client_weights_mode == "sample_size"
            and hasattr(self, "client_sample_size")
            and client_id in self.client_sample_size
        ):
            weight = self.client_sample_size[client_id] / sum(
                self.client_sample_size.values()
            )
        else:
            weight = 1.0 / self.aggregator_configs.get("num_clients", 1)
        alpha_t = (
            self.alpha
            * self.staleness_fn(self.global_step - self.client_step[client_id])
            * weight
        )

        for name in self.global_state:
            # Skip integer parameters by averaging them later in the `aggregate` method
            if (
                (
                    self.named_parameters is not None
                    and name not in self.named_parameters
                )
                or self.global_state[name].dtype == torch.int64
                or self.global_state[name].dtype == torch.int32
            ):
                continue
            self.step[name] = (
                alpha_t * (-local_model[name])
                if gradient_based
                else alpha_t * (local_model[name] - self.global_state[name])
            )

    def __staleness_fn_factory(self, staleness_fn_name, **kwargs):
        if staleness_fn_name == "constant":
            return lambda u: 1
        elif staleness_fn_name == "polynomial":
            a = kwargs["a"]
            return lambda u: (u + 1) ** (-a)
        elif staleness_fn_name == "hinge":
            a = kwargs["a"]
            b = kwargs["b"]
            return lambda u: 1 if u <= b else 1.0 / (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
