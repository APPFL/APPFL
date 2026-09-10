"""
DecentralizedDLoRABSVDAggregator: Server-side aggregator implementing
decentralized CLIP + LoRA fine-tuning via topology-aware AB_SVD aggregation.

Combines the topology consensus logic from DecentralizedAggregator with the
AB_SVD LoRA aggregation from DLoRABSVDAggregator.

For each client's consensus neighborhood (defined by the topology):
  LoRA pairs (w_lora_A, w_lora_B):
    1. Compute the topology-weighted average of B @ A across neighbors.
    2. Truncated SVD:  (B@A)_avg ≈ U_r * diag(S_r) * Vh_r
    3. Symmetric split:
         new_B = U_r  * sqrt(S_r)    shape: (out_dim, r)
         new_A = sqrt(S_r) * Vh_r    shape: (r, in_dim)
  All other parameters: standard topology-weighted average.

Topologies:
  "fc"   — Fully connected: all N clients form a single neighborhood.
            Each client receives the same AB_SVD-aggregated model.
            Mixing weight = 1/N (equal) or proportional to sample size.
  "ring" — Ring: each client's neighborhood = {left, self, right}.
            Each client receives its own AB_SVD model from its 3 neighbors.
            Mixing weight = 1/3 for each neighbor (standard ring semantics).

Returns:
    Dict[client_id -> per-client state dict] for both topologies,
    matching the decentralized aggregator's interface.

References:
    dLoRA AB_SVD: clip_lora/dlora_utils/fed_global.py (fedavg_AB_SVD)
    Decentralized consensus: CDSGD / SGP (Assran et al., ICML 2019)
"""

import copy
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from appfl.algorithm.aggregator import BaseAggregator


class DecentralizedDLoRABSVDAggregator(BaseAggregator):
    """
    Decentralized LoRA aggregator: topology-aware consensus with AB_SVD.

    aggregator_kwargs (YAML config):
        topology (str):
            "fc"   – fully connected (default).
            "ring" – ring with 2 nearest neighbors.
        lora_rank (int):
            SVD truncation rank. Must match the LoRA rank r used by all
            clients. Default: 2.
        client_weights_mode (str):
            "equal"       – uniform mixing weight (default for ring; also
                            applies to FC when "sample_size" data are absent).
            "sample_size" – weight proportional to dataset size (FC only;
                            ring always uses equal weights of 1/3).
        device (str):
            Device for SVD computation. Default: "cuda" if available else "cpu".
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

        self.topology = aggregator_configs.get("topology", "fc")
        self.lora_rank = aggregator_configs.get("lora_rank", 2)
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )
        _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = aggregator_configs.get("device", _default_device)

        # Per-client sample sizes (populated by set_sample_size calls)
        self.sample_sizes: Dict[str, int] = {}

        # Representative global state (first client's model for get_parameters)
        self.global_state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # BaseAggregator interface
    # ------------------------------------------------------------------

    def get_parameters(self, **kwargs) -> Dict:
        """Return a representative global state dict."""
        if self.global_state is not None:
            return self.global_state
        if self.model is not None:
            return self.model.state_dict()
        raise ValueError(
            "DecentralizedDLoRABSVDAggregator has no model or global state."
        )

    def aggregate(self, local_models: Dict, **kwargs) -> Dict[str, Dict]:
        """
        Aggregate local LoRA models using topology-aware AB_SVD consensus.

        Args:
            local_models: dict mapping client_id -> full model state_dict.

        Returns:
            Dict mapping client_id -> per-client aggregated state dict.
            FC: all values are identical (global AB_SVD model).
            Ring: each value is the AB_SVD model for that client's neighborhood.
        """
        if self.topology == "ring":
            result = self._aggregate_ring(local_models)
        else:
            result = self._aggregate_fc(local_models)

        # Store first client's model as the representative global state
        first_cid = next(iter(result))
        self.global_state = result[first_cid]

        return result

    # ------------------------------------------------------------------
    # Topology implementations
    # ------------------------------------------------------------------

    def _aggregate_fc(self, local_models: Dict) -> Dict[str, Dict]:
        """
        Fully-connected consensus: AB_SVD over all client models.
        All clients receive the same aggregated LoRA model.
        """
        client_ids = list(local_models.keys())
        weights = self._fc_weights(client_ids)

        averaged_state = self._ab_svd_aggregate(local_models, weights)

        if self.logger:
            self.logger.info(
                f"[Decentralized-DLoRA-FC] AB_SVD over {len(client_ids)} clients "
                f"(lora_rank={self.lora_rank}, weights_mode={self.client_weights_mode})"
            )

        return {cid: copy.deepcopy(averaged_state) for cid in client_ids}

    def _aggregate_ring(self, local_models: Dict) -> Dict[str, Dict]:
        """
        Ring topology consensus: each client gets the AB_SVD aggregate of
        its own model plus its two ring neighbors (weight = 1/3 each).
        """
        # Sort so the ring order is deterministic across rounds regardless
        # of which client sends its update first.
        client_ids = sorted(local_models.keys())
        num_agents = len(client_ids)
        pi = self._build_ring_pi(num_agents)  # list of {j_idx: weight}

        per_client_states: Dict[str, Dict] = {}
        for agent_idx, cid in enumerate(client_ids):
            neighbor_weights_idx = pi[agent_idx]  # {j_idx: 1/3}

            # Build neighborhood sub-dict with string client_id keys
            neighborhood: Dict[str, Dict] = {}
            nbr_weights: Dict[str, float] = {}
            for j_idx, w in neighbor_weights_idx.items():
                nbr_cid = client_ids[j_idx]
                neighborhood[nbr_cid] = local_models[nbr_cid]
                nbr_weights[nbr_cid] = w

            per_client_states[cid] = self._ab_svd_aggregate(neighborhood, nbr_weights)

            if self.logger:
                nbr_names = [client_ids[j] for j in neighbor_weights_idx]
                self.logger.info(
                    f"[Decentralized-DLoRA-Ring] Agent {cid} AB_SVD with "
                    f"neighbors {nbr_names} (lora_rank={self.lora_rank})"
                )

        return per_client_states

    # ------------------------------------------------------------------
    # Core AB_SVD aggregation
    # ------------------------------------------------------------------

    def _ab_svd_aggregate(
        self,
        local_models: Dict[str, Dict],
        weights: Dict[str, float],
    ) -> Dict:
        """
        AB_SVD aggregation over a set of local models with given weights.

        For w_lora_A / w_lora_B pairs:
            weighted_product = sum_i( weights[i] * B_i @ A_i )
            U_r, S_r, Vh_r  = truncated_SVD(weighted_product.T, r=lora_rank)
            new_B = U_r * sqrt(S_r)   (in_dim, r) -> transpose matches B shape
            new_A = sqrt(S_r) * Vh_r  (r, out_dim)

        For all other parameters: weighted average (FedAvg).
        """
        client_ids = list(local_models.keys())
        first_state = local_models[client_ids[0]]

        global_dict: Dict = {}
        written_lora_b_keys = set()

        for key in first_state.keys():
            # w_lora_B keys are handled together with their paired w_lora_A key
            if "w_lora_B" in key:
                continue

            if "w_lora_A" in key:
                layer_prefix = key.replace("w_lora_A", "")
                key_A = f"{layer_prefix}w_lora_A"
                key_B = f"{layer_prefix}w_lora_B"

                # Weighted sum of B @ A products across neighborhood
                summed_product = None
                for cid in client_ids:
                    A = local_models[cid][key_A]
                    B = local_models[cid][key_B]
                    product = B @ A  # (out_dim, in_dim)
                    scaled = product * weights[cid]
                    if summed_product is None:
                        summed_product = scaled.clone()
                    else:
                        summed_product = summed_product + scaled

                summed_product = summed_product.to(self.device)

                # Truncated SVD on the transposed product
                # summed_product.t() shape: (in_dim, out_dim)
                U, S, Vh = torch.linalg.svd(summed_product.t(), full_matrices=False)
                r = self.lora_rank
                U_r = U[:, :r]  # (in_dim, r)
                S_r = S[:r]  # (r,)
                Vh_r = Vh[:r, :]  # (r, out_dim)
                sqrt_S_r = torch.sqrt(S_r)

                # Symmetric split — shapes match original A and B
                # new_B: (out_dim, r)  →  (U_r * sqrt_S_r).T then T again
                #   Following the same convention as DLoRABSVDAggregator:
                #   new_B = U_r * sqrt_S_r  (in_dim, r) — broadcast col-wise
                #   new_A = sqrt_S_r[:,None] * Vh_r     (r, out_dim)
                new_B = U_r * sqrt_S_r.unsqueeze(0)  # (in_dim, r)
                new_A = sqrt_S_r.unsqueeze(1) * Vh_r  # (r, out_dim)

                global_dict[key_B] = new_B.cpu()
                global_dict[key_A] = new_A.cpu()
                written_lora_b_keys.add(key_B)

            else:
                # Standard weighted average for non-LoRA parameters
                param = first_state[key]
                if param.dtype in (torch.int32, torch.int64, torch.bool):
                    # Integer/bool buffers: take from first client unchanged
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fc_weights(self, client_ids: List[str]) -> Dict[str, float]:
        """Compute per-client mixing weights for FC topology."""
        if self.client_weights_mode == "sample_size" and self.sample_sizes:
            total = sum(self.sample_sizes.get(cid, 1) for cid in client_ids)
            return {cid: self.sample_sizes.get(cid, 1) / total for cid in client_ids}
        w = 1.0 / len(client_ids)
        return {cid: w for cid in client_ids}

    @staticmethod
    def _build_ring_pi(num_agents: int) -> List[Dict[int, float]]:
        """
        Build ring mixing weights.
        Returns list of dicts: pi[i] = {j_idx: 1/3} for j in {i-1, i, i+1}.
        """
        pi = []
        for i in range(num_agents):
            neighbors = [
                (i - 1) % num_agents,
                i,
                (i + 1) % num_agents,
            ]
            pi.append({j: 1.0 / len(neighbors) for j in neighbors})
        return pi
