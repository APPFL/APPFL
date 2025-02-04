import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional
import numpy as np
import appfl.misc.data_readiness as dr

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
        logger: Optional[Any] = None
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get("client_weights_mode")
        

        if self.model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None # Models parameters that are used for aggregation, this is unknown at the beginning

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

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs) -> Dict:

        """
        Take the weighted average of local models from clients and return the global model.
        """
        if self.global_state is None:
            if self.model is not None:
                try: 
                    self.global_state = {
                        name: self.model.state_dict()[name] for name in list(local_models.values())[0]
                    }
                except:
                    self.global_state = {
                        name: tensor.detach().clone() for name, tensor in list(local_models.values())[0].items()
                    }
            else:
                self.global_state = {
                    name: tensor.detach().clone() for name, tensor in list(local_models.values())[0].items()
                }
        
        self.compute_steps(local_models, **kwargs)
        
        for name in self.global_state:
            if name in self.step:
                self.global_state[name] = self.global_state[name] + self.step[name]
            else:
                param_sum = torch.zeros_like(self.global_state[name])
                for _, model in local_models.items():
                    param_sum += model[name]
                self.global_state[name] = torch.div(param_sum, len(local_models))
        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)
        return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs):
        """
        Compute the changes to the global model after the aggregation.
        """ 

        # metric_mappings = {}
        
        # for id_, metric_list in kwargs['metrics'].items():
        #     metric_mappings[id_] = {metric: kwargs[metric][id_] for metric in metric_list}



        # # client_class_imbalance = kwargs.get("class_imbalance", {})
        # # client_class_imbalance = {k: v for k, v in client_class_imbalance.items() if v is not None}

        # # client_variance = kwargs.get("variance", {})
        # # client_variance = {k: v for k, v in client_variance.items() if v is not None}
        

        for name in self.global_state:
            if self.named_parameters is not None and name not in self.named_parameters:
                continue
            self.step[name] = torch.zeros_like(self.global_state[name])
        
        for client_id, model in local_models.items():
            if (
                self.client_weights_mode == "sample_size" and
                hasattr(self, "client_sample_size") and
                client_id in self.client_sample_size
            ):
                weight = self.client_sample_size[client_id] / sum(self.client_sample_size.values())
            elif self.client_weights_mode == "data_ready":

                brisque_scores = kwargs.get("brisque", {})
                # Min-Max Normalize BRISQUE scores
                min_brisque = min(brisque_scores.values())
                max_brisque = max(brisque_scores.values())

                if max_brisque == min_brisque:  # Edge case: all scores are the same
                    normalized_brisque = {client: 0.5 for client in brisque_scores}
                else:
                    normalized_brisque = {
                        client: (score - min_brisque) / (max_brisque - min_brisque)
                        for client, score in brisque_scores.items()
                    }

                # Apply exponential decay weighting
                alpha = 10  # Controls sensitivity of weighting
                raw_weights = {client: np.exp(-alpha * norm_score) for client, norm_score in normalized_brisque.items()}

                # Normalize weights to sum to 1
                total_weight = sum(raw_weights.values())
                weights = {client: raw_weights[client] / total_weight for client in raw_weights}

                weight = raw_weights[client_id]

                print(f"Client {client_id} weight: {weight}")

            #     distances = dr.find_most_diverse_client(metric_mappings)
            #     weights_mapping = dr.compute_client_weights(distances)

            #     if client_id in weights_mapping:
            #         weight = weights_mapping[client_id]
            #     else:
            #         raise ValueError(f"Weight for client {client_id} is missing in weights_mapping.")

                # Print or log the weight for debugging purposes
                # print(f"Client {client_id} weight: {weight}")
            else:
                weight = 1.0 / len(local_models)
              
            for name in model:
                if name in self.step:
                    self.step[name] += weight * (model[name] - self.global_state[name])


# def _calculate_weight(
#     sample_size_prop=None,
#     client_class_imbalance=None,
#     client_variance=None,
#     variance_threshold=100.0,
#     default_weight=0.5
# ):
#     weights = []
#     total_weight = 0

#     # Sample size factor
#     if sample_size_prop is not None:
#         sample_size_weight = 0
#         weights.append((sample_size_weight, sample_size_prop))
#         total_weight += sample_size_weight

#     # Imbalance factor
#     if client_class_imbalance is not None:
#         imbalance_weight = 1
#         if client_class_imbalance == np.inf:
#             imbalance_factor = 0
#         else:
#             imbalance_factor = 1.0 - client_class_imbalance
#         weights.append((imbalance_weight, imbalance_factor))
#         total_weight += imbalance_weight

#     # Variance factor
#     if client_variance is not None:
#         variance_weight = 1
#         variance_factor = np.exp(-client_variance / variance_threshold)
#         weights.append((variance_weight, variance_factor))
#         total_weight += variance_weight

#     # Calculate combined weight
#     if weights:
#         combined_weight = sum(weight * factor for weight, factor in weights) / total_weight
#     else:
#         combined_weight = default_weight

#     # Ensure the weight is not zero
#     return max(combined_weight, 0.000001)