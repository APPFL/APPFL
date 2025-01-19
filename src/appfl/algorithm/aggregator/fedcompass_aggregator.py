import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class FedCompassAggregator(BaseAggregator):
    """
    FedCompass asynchronous federated learning algorithm.
    For more details, check paper: https://arxiv.org/abs/2309.14675
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
        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name=self.aggregator_configs.get("staleness_fn", "constant"),
            **self.aggregator_configs.get("staleness_fn_kwargs", {}),
        )
        self.alpha = self.aggregator_configs.get("alpha", 0.9)

        self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        if model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

    def aggregate(
        self,
        client_id: Optional[Union[str, int]] = None,
        local_model: Optional[Union[Dict, OrderedDict]] = None,
        local_models: Optional[Dict[Union[str, int], Union[Dict, OrderedDict]]] = None,
        staleness: Optional[Union[int, Dict[Union[str, int], int]]] = None,
        **kwargs,
    ) -> Dict:
        if self.global_state is None:
            if client_id is not None and local_model is not None:
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
            else:
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

        gradient_based = self.aggregator_configs.get("gradient_based", False)
        if client_id is not None and local_model is not None:
            weight = 1.0 / self.aggregator_configs.get("num_clients", 1)
            alpha_t = self.alpha * self.staleness_fn(staleness) * weight

            for name in self.global_state:
                if (
                    self.named_parameters is not None
                    and name not in self.named_parameters
                ) or (
                    self.global_state[name].dtype == torch.int64
                    or self.global_state[name].dtype == torch.int32
                ):
                    self.global_state[name] = local_model[name]
                else:
                    if gradient_based:
                        self.global_state[name] = (
                            self.global_state[name] - local_model[name] * alpha_t
                        )
                    else:
                        self.global_state[name] = (1 - alpha_t) * self.global_state[
                            name
                        ] + local_model[name] * alpha_t

        else:
            if not gradient_based:
                global_state_cp = copy.deepcopy(self.global_state)
                for name in global_state_cp:
                    global_state_cp[name] = torch.zeros_like(global_state_cp[name])
            alpha_t_sum = 0
            for i, client_id in enumerate(local_models):
                local_model = local_models[client_id]
                weight = 1.0 / self.aggregator_configs.get("num_clients", 1)
                alpha_t = self.alpha * self.staleness_fn(staleness[client_id]) * weight
                alpha_t_sum += alpha_t
                for name in self.global_state:
                    if (
                        self.named_parameters is not None
                        and name not in self.named_parameters
                    ) or (
                        self.global_state[name].dtype == torch.int64
                        or self.global_state[name].dtype == torch.int32
                    ):
                        if i == 0:
                            self.global_state[name] = torch.zeros_like(
                                local_model[name]
                            )
                        self.global_state[name] = (
                            self.global_state[name] + local_model[name]
                        )
                        if i == len(local_models) - 1:
                            self.global_state[name] = torch.div(
                                self.global_state[name], len(local_models)
                            ).type(self.global_state[name].dtype)
                    else:
                        if gradient_based:
                            self.global_state[name] = (
                                self.global_state[name] - local_model[name] * alpha_t
                            )
                        else:
                            global_state_cp[name] += local_model[name] * alpha_t
                            if i == len(local_models) - 1:
                                self.global_state[name] = (
                                    1 - alpha_t_sum
                                ) * self.global_state[name] + global_state_cp[name]

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)
        return {k: v.clone() for k, v in self.global_state.items()}

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
