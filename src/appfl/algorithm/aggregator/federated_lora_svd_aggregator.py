"""
FederatedLoRASVDAggregator: Federated aggregator for CLIP + LoRA fine-tuning
using AB_SVD aggregation.

For LoRA parameter pairs (w_lora_A, w_lora_B):
    1. Compute the weighted average of B @ A across all clients.
    2. Truncated SVD:  (B@A)_avg ≈ U_r * diag(S_r) * Vh_r
    3. Symmetric split:
         new_B = U_r  * sqrt(S_r)    shape: (in_dim, r)
         new_A = sqrt(S_r) * Vh_r    shape: (r, out_dim)
All other parameters: standard weighted average (FedAvg).

Returns a single global state dict broadcast to all clients.
"""

import copy
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from appfl.algorithm.aggregator import BaseAggregator


class FederatedLoRASVDAggregator(BaseAggregator):
    """
    Federated LoRA aggregator using AB_SVD decomposition.

    Aggregates LoRA adapters from all clients into a single global model
    via truncated SVD on the weighted-average low-rank product B @ A.
    All other (non-LoRA) parameters are averaged with FedAvg.

    aggregator_kwargs (YAML config):
        lora_rank (int):
            SVD truncation rank. Must match the LoRA rank r used by all
            clients. Default: 2.
        client_weights_mode (str):
            "equal"       – uniform mixing weight (default).
            "sample_size" – weight proportional to dataset size.
        device (str):
            Device for SVD computation. Default: "cpu".
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

        self.lora_rank = aggregator_configs.get("lora_rank", 2)
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )
        self.device = aggregator_configs.get("device", "cpu")

        self.sample_sizes: Dict[str, int] = {}
        self.global_state: Optional[Dict] = None

    def get_parameters(self, **kwargs) -> Dict:
        if self.global_state is not None:
            return copy.deepcopy(self.global_state)
        if self.model is not None:
            return copy.deepcopy(self.model.state_dict())
        raise ValueError("FederatedLoRASVDAggregator has no model or global state.")

    def aggregate(self, local_models: Dict, **kwargs) -> Dict:
        """
        Aggregate local LoRA models using AB_SVD.

        Args:
            local_models: dict mapping client_id -> LoRA state dict.

        Returns:
            Single aggregated state dict broadcast to all clients.
        """
        client_ids = list(local_models.keys())
        weights = self._compute_weights(client_ids)

        self.global_state = self._ab_svd_aggregate(local_models, weights)

        if self.logger:
            self.logger.info(
                f"[FederatedLoRASVD] AB_SVD aggregation over {len(client_ids)} clients "
                f"(lora_rank={self.lora_rank}, weights_mode={self.client_weights_mode})"
            )

        return copy.deepcopy(self.global_state)

    def _compute_weights(self, client_ids: List[str]) -> Dict[str, float]:
        if self.client_weights_mode == "sample_size" and self.sample_sizes:
            total = sum(self.sample_sizes.get(cid, 1) for cid in client_ids)
            return {cid: self.sample_sizes.get(cid, 1) / total for cid in client_ids}
        w = 1.0 / len(client_ids)
        return {cid: w for cid in client_ids}

    def _ab_svd_aggregate(
        self,
        local_models: Dict[str, Dict],
        weights: Dict[str, float],
    ) -> Dict:
        """
        AB_SVD aggregation over all client models.

        For w_lora_A / w_lora_B pairs:
            weighted_product = sum_i( weights[i] * B_i @ A_i )
            U_r, S_r, Vh_r  = truncated_SVD(weighted_product.T, r=lora_rank)
            new_B = U_r * sqrt(S_r)   (in_dim, r)
            new_A = sqrt(S_r) * Vh_r  (r, out_dim)

        For all other parameters: weighted average.
        """
        client_ids = list(local_models.keys())
        first_state = local_models[client_ids[0]]

        global_dict: Dict = {}
        written_lora_b_keys = set()

        for key in first_state.keys():
            if "w_lora_B" in key:
                continue

            if "w_lora_A" in key:
                layer_prefix = key.replace("w_lora_A", "")
                key_A = f"{layer_prefix}w_lora_A"
                key_B = f"{layer_prefix}w_lora_B"

                summed_product = None
                for cid in client_ids:
                    A = local_models[cid][key_A]
                    B = local_models[cid][key_B]
                    scaled = (B @ A) * weights[cid]  # (out_dim, in_dim)
                    summed_product = (
                        scaled.clone()
                        if summed_product is None
                        else summed_product + scaled
                    )

                summed_product = summed_product.to(self.device)

                U, S, Vh = torch.linalg.svd(summed_product.t(), full_matrices=False)
                r = self.lora_rank
                U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
                sqrt_S_r = torch.sqrt(S_r)

                global_dict[key_B] = (U_r * sqrt_S_r.unsqueeze(0)).cpu()  # (in_dim, r)
                global_dict[key_A] = (
                    sqrt_S_r.unsqueeze(1) * Vh_r
                ).cpu()  # (r, out_dim)
                written_lora_b_keys.add(key_B)

            else:
                param = first_state[key]
                if param.dtype in (torch.int32, torch.int64, torch.bool):
                    global_dict[key] = param.clone()
                else:
                    aggregated = sum(
                        local_models[cid][key].float() * weights[cid]
                        for cid in client_ids
                    )
                    global_dict[key] = aggregated.to(param.dtype).cpu()

        # Safety pass: handle any w_lora_B keys not yet written
        for key in first_state.keys():
            if (
                "w_lora_B" in key
                and key not in written_lora_b_keys
                and key not in global_dict
            ):
                param = first_state[key]
                aggregated = sum(
                    local_models[cid][key].float() * weights[cid] for cid in client_ids
                )
                global_dict[key] = aggregated.to(param.dtype).cpu()

        return global_dict
