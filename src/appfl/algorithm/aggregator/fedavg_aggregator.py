import gc
import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional
from appfl.misc.memory_utils import (
    clone_state_dict_optimized,
    safe_inplace_operation,
    optimize_memory_cleanup,
)


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

        # Check for optimize_memory in aggregator_configs, default to True
        self.optimize_memory = getattr(aggregator_configs, "optimize_memory", True)

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
                if self.optimize_memory:
                    return clone_state_dict_optimized(self.model.state_dict())
                else:
                    return copy.deepcopy(self.model.state_dict())
            else:
                raise ValueError("Model is not provided to the aggregator.")

        if self.optimize_memory:
            return clone_state_dict_optimized(self.global_state)
        else:
            return {k: v.clone() for k, v in self.global_state.items()}

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.

        Supports streamed aggregation: if _chunk_idx is in kwargs, only aggregates that chunk.
        """
        # Check if this is streamed aggregation
        self.logger.info(f"Kwargs received in aggregate: {kwargs}")
        if "_chunk_idx" in kwargs:
            return self._aggregate_chunk(local_models, **kwargs)
        # Memory optimization: Initialize global state efficiently
        if self.global_state is None:
            if self.model is not None:
                try:
                    if self.optimize_memory:
                        # More memory-efficient initialization
                        self.global_state = {}
                        model_state = self.model.state_dict()
                        first_model = list(local_models.values())[0]
                        with torch.no_grad():
                            for name in first_model:
                                if name in model_state:
                                    self.global_state[name] = (
                                        model_state[name].clone().detach()
                                    )
                        gc.collect()
                    else:
                        self.global_state = {
                            name: self.model.state_dict()[name]
                            for name in list(local_models.values())[0]
                        }
                except:  # noqa E722
                    if self.optimize_memory:
                        self.global_state = {}
                        with torch.no_grad():
                            for name, tensor in list(local_models.values())[0].items():
                                self.global_state[name] = tensor.detach().clone()
                        gc.collect()
                    else:
                        self.global_state = {
                            name: tensor.detach().clone()
                            for name, tensor in list(local_models.values())[0].items()
                        }
            else:
                if self.optimize_memory:
                    self.global_state = {}
                    with torch.no_grad():
                        for name, tensor in list(local_models.values())[0].items():
                            self.global_state[name] = tensor.detach().clone()
                    gc.collect()
                else:
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in list(local_models.values())[0].items()
                    }

        self.compute_steps(local_models)

        # Memory optimization: More efficient aggregation with cleanup
        if self.optimize_memory:
            with torch.no_grad():
                for name in self.global_state:
                    if name in self.step:
                        # Use safe in-place operations with dtype checking
                        self.global_state[name] = safe_inplace_operation(
                            self.global_state[name], "add", self.step[name]
                        )
                    else:
                        param_sum = torch.zeros_like(self.global_state[name])
                        # Efficiently sum parameters with dtype checking
                        for _, model in local_models.items():
                            param_sum = safe_inplace_operation(
                                param_sum, "add", model[name]
                            )

                        # Safe division with dtype handling
                        self.global_state[name] = safe_inplace_operation(
                            param_sum, "div", len(local_models)
                        )
                        optimize_memory_cleanup(param_sum, force_gc=False)

            optimize_memory_cleanup(force_gc=True)
            self.step.clear()
        else:
            # Original behavior
            for name in self.global_state:
                if name in self.step:
                    self.global_state[name] = self.global_state[name] + self.step[name]
                else:
                    param_sum = torch.zeros_like(self.global_state[name])
                    for _, model in local_models.items():
                        param_sum += model[name]
                    # make sure global state have the same type as the local model
                    self.global_state[name] = torch.div(
                        param_sum, len(local_models)
                    ).type(param_sum.dtype)

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        if self.optimize_memory:
            return clone_state_dict_optimized(self.global_state)
        else:
            return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        # Memory optimization: More efficient step computation
        if self.optimize_memory:
            with torch.no_grad():
                for name in self.global_state:
                    # Skip integer parameters by averaging them later in the `aggregate` method
                    if (
                        self.named_parameters is not None
                        and name not in self.named_parameters
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
                            # Safe in-place gradient accumulation
                            diff = model[name] - self.global_state[name]
                            weighted_diff = diff * weight
                            self.step[name] = safe_inplace_operation(
                                self.step[name], "add", weighted_diff
                            )
                            optimize_memory_cleanup(diff, weighted_diff, force_gc=False)
        else:
            # Original behavior
            for name in self.global_state:
                # Skip integer parameters by averaging them later in the `aggregate` method
                if (
                    self.named_parameters is not None
                    and name not in self.named_parameters
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
                        self.step[name] += weight * (
                            model[name] - self.global_state[name]
                        )

    def _aggregate_chunk(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        """Memory-efficient chunk aggregation for streamed aggregation."""
        # Extract chunk metadata (scheduler aggregates kwargs by client_id)
        # All clients should have same chunk_idx/keys/total, so take from first client
        chunk_keys_dict = kwargs["_chunk_keys"]
        chunk_keys = list(chunk_keys_dict.values())[0]  # Get from first client

        self.logger.info(
            f"Aggregating chunk with keys: {chunk_keys} from {len(local_models)} clients."
        )

        # Initialize global state for chunk if needed
        if self.global_state is None:
            self.global_state = {}

        with torch.no_grad():
            first_model = list(local_models.values())[0]
            for key in chunk_keys:
                if key not in self.global_state and key in first_model:
                    self.global_state[key] = first_model[key].detach().clone()

        # Compute and apply aggregation for chunk
        self._compute_chunk_steps(local_models, chunk_keys)

        with torch.no_grad():
            for key in chunk_keys:
                if key in self.step:
                    self.global_state[key] = safe_inplace_operation(
                        self.global_state[key], "add", self.step[key]
                    )
                else:
                    param_sum = torch.zeros_like(self.global_state[key])
                    for model in local_models.values():
                        param_sum = safe_inplace_operation(param_sum, "add", model[key])
                    self.global_state[key] = safe_inplace_operation(
                        param_sum, "div", len(local_models)
                    )
                    optimize_memory_cleanup(param_sum, force_gc=False)

        optimize_memory_cleanup(force_gc=True)
        self.step.clear()

        # Update model (partial)
        if self.model is not None:
            current_state = self.model.state_dict()
            for key in chunk_keys:
                if key in self.global_state:
                    current_state[key] = self.global_state[key]
            self.model.load_state_dict(current_state, strict=False)

        # Return aggregated chunk
        return clone_state_dict_optimized({k: self.global_state[k] for k in chunk_keys})

    def _compute_chunk_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], chunk_keys: list
    ):
        """Compute aggregation steps for chunk parameters."""
        with torch.no_grad():
            for key in chunk_keys:
                if key in self.global_state:
                    if (
                        self.global_state[key].dtype == torch.int64
                        or self.global_state[key].dtype == torch.int32
                    ):
                        continue
                    self.step[key] = torch.zeros_like(self.global_state[key])

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

                for key in chunk_keys:
                    if key in self.step and key in model:
                        diff = model[key] - self.global_state[key]
                        weighted_diff = diff * weight
                        self.step[key] = safe_inplace_operation(
                            self.step[key], "add", weighted_diff
                        )
                        optimize_memory_cleanup(diff, weighted_diff, force_gc=False)
