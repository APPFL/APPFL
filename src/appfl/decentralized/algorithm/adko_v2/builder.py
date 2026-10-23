"""Build an :class:`ADKOv2Agent` from configuration -- the ADKO v2 half of config loading.

The counterpart of ``adko/builder.py``, with the same signature, so a launcher builds either
algorithm the same way. The generic half (graph, rounds, communication budget) is
:mod:`appfl.decentralized.config`; this module reads the keys only v2 has.

Schema, under ``algorithm_configs`` in a node config; every default is the settled method's::

    weights:        {beta: 2, lam: 2, gamma: 2, kernel: euclidean|hamming,
                     bandwidth: null (1 for euclidean, 0.5 for hamming), eps: 0.05}
    q_bits:         8           # 1, 2, 3, 4, 8 or 32
    memory_rounds:  3           # FIFO memory = memory_rounds x degree tokens
    trust:          true        # odds-weighted; false weighs peers equally
    trust_eta:      0.1
    warmup_rounds:  5
    objective:      maximize

and, optionally, the L3 planner under ``llm_configs`` (see :func:`create_planner`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from appfl.decentralized.config import as_kwargs, load_callable
from appfl.decentralized.metrics import Meter, metered_evaluator
from appfl.decentralized.neighbor import Neighbors
from appfl.decentralized.topology import Topology

from appfl.decentralized.algorithm.adko_v2.agent import ADKOv2Agent
from appfl.decentralized.algorithm.adko_v2.components import Planner
from appfl.decentralized.algorithm.adko_v2.llm import build_policy
from appfl.decentralized.algorithm.adko_v2.reasoning import ReasoningWeights

__all__ = ["create_agent", "create_planner", "create_weights"]

#: Keys of ``llm_configs``: the reference's ``src.v2.run`` flag names, plus where to write.
LLM_KEYS = (
    "llm",  # none | l3 | random (the free control)
    "alpha",
    "llm_model",  # "mock" runs the offline mock client
    "llm_effort",
    "llm_share",  # none | hypothesis | note | reasoning
    "llm_solo",
    "llm_base_url",  # an OpenAI-compatible chat server; omitted: the Responses API
    # The key for llm_base_url's server, better left to $APPFL_DECENTRALIZED_LLM_API_KEY;
    # the Responses API (no llm_base_url) reads $OPENAI_API_KEY itself.
    "llm_api_key",
    "llm_max_tokens",
    # Per lab, since each lab owns its planner; the reference caps a whole run.
    "max_usd",
    "label",  # names the run's call ids, as the reference's --label does; "ADKO-v2"
    "run_dir",  # the call cache and each lab's call log; default as the reference lays it out
)


def create_weights(config: Optional[DictConfig]) -> ReasoningWeights:
    """The score's weights: the settled defaults, then any field the config sets."""
    fields = {k: v for k, v in as_kwargs(config).items() if v is not None}
    unknown = set(fields) - set(ReasoningWeights.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown ADKO v2 weights field(s) {sorted(unknown)}")
    for key in ("beta", "lam", "gamma", "bandwidth", "eps"):
        if key in fields:
            fields[key] = float(fields[key])
    return ReasoningWeights(**fields)


def create_planner(
    llm_configs: Optional[Dict[str, Any]],
    *,
    agent_id: str,
    seed: int,
    n_labs: int,
    setting: Optional[str],
    warmup_rounds: int,
) -> Optional[Planner]:
    """One agent's L3 planner from an ``llm_configs`` block, or ``None`` (the LM-free method).

    The block takes the reference's flag names (:data:`LLM_KEYS`), so an L3 arm here is
    configured with the same words as there::

        llm_configs:
          llm: l3
          alpha: 0.3
          llm_model: Qwen/Qwen3.6-35B-A3B-FP8
          llm_base_url: http://localhost:8000/v1
          llm_max_tokens: 16000
          label: ADKO-v2f-odds-bit-L3-a0.3

    Without ``run_dir`` a run writes to ``./output/l3/{setting}/{label}/seed{seed}_llm``, the
    reference's layout, so runs of different seeds or arms never share a log.
    """
    fields = dict(llm_configs or {})
    unknown = set(fields) - set(LLM_KEYS)
    if unknown:
        raise ValueError(f"unknown llm_configs field(s) {sorted(unknown)}")
    mode = str(fields.get("llm") or "none")
    if mode == "none":
        return None
    if mode not in ("l3", "random"):
        raise ValueError(f"llm must be none, l3 or random, got {mode!r}")
    fields["label"] = (
        fields.get("label") or "ADKO-v2"
    )  # the reference's default --label
    run_dir = (
        fields.pop("run_dir", None)
        or f"./output/l3/{setting}/{fields['label']}/seed{seed}_llm"
    )
    fields.update(
        seed=seed,
        setting=setting,
        agents=n_labs,
        warmup_rounds=warmup_rounds,
        llm_api_key=fields.get("llm_api_key")
        or os.environ.get("APPFL_DECENTRALIZED_LLM_API_KEY"),
        log_name=f"llm_calls_{agent_id}.jsonl",
    )
    return build_policy(fields, run_dir)


def create_agent(
    node_config: DictConfig,
    neighbors: Neighbors,
    meter: Meter,
    llm_config: Optional[Dict[str, Any]] = None,
    topology: Optional[Topology] = None,
) -> ADKOv2Agent:
    """Build one agent from one node's config: once per node, whatever the transport.

    :param neighbors: who this node exchanges with. Its size is the agent's degree, which sets
        the memory budget.
    :param llm_config: overrides for the ``llm_configs`` block, keyed as in :data:`LLM_KEYS`.
    :param topology: the whole graph, when a simulation launcher has one. Supplies the node's
        index and its neighbours' indices. Without it the config must give ``node_index``:
        the index is the ``source`` of every token the node sends, so two nodes sharing one
        would discard each other's tokens as their own.
    """
    algorithm = node_config.algorithm_configs

    agent_id = str(node_config.node_id)
    if topology is not None:
        roster = list(topology.node_ids)
        index = roster.index(agent_id)
        neighbor_indices = [roster.index(peer) for peer in neighbors.recv_from]
        n_labs = len(roster)
    else:
        if node_config.get("node_index", None) is None:
            raise ValueError(
                f"{agent_id}: a node without a topology needs `node_index`, its position in "
                f"the federation -- the source id its tokens carry"
            )
        index = int(node_config.node_index)
        neighbor_indices = []  # learned from the first tokens each neighbour sends
        n_labs = None
    # The run seed, not offset by the index: every private stream is keyed on
    # (seed, index) inside the agent, which is how the reference keys them.
    seed = int(node_config.get("seed", 0))

    space_configs = node_config.space_configs
    space_kwargs = as_kwargs(space_configs.get("space_kwargs", None))
    if int(space_kwargs.get("agent_index", index)) != index:
        raise ValueError(
            f"{agent_id}: space_kwargs.agent_index is {space_kwargs['agent_index']} but the "
            f"node's index is {index}; they must agree, or the node's slice and warmup are "
            f"another node's"
        )
    space = load_callable(space_configs.space_path, space_configs.space_name)(
        **space_kwargs
    )
    if n_labs is None:
        n_labs = int(space_kwargs.get("num_agents", len(neighbors.recv_from) + 1))

    surrogate_configs = node_config.surrogate_configs
    surrogate = load_callable(
        surrogate_configs.surrogate_path, surrogate_configs.surrogate_name
    )(**as_kwargs(surrogate_configs.get("surrogate_kwargs", None)))

    evaluator_configs = node_config.evaluator_configs
    evaluator = load_callable(
        evaluator_configs.evaluator_path, evaluator_configs.evaluator_name
    )(**as_kwargs(evaluator_configs.get("evaluator_kwargs", None)))
    evaluator = metered_evaluator(
        evaluator, meter, cost_seconds=float(evaluator_configs.get("cost_seconds", 0.0))
    )

    warmup_rounds = int(algorithm.get("warmup_rounds", 5))
    llm_configs = dict(as_kwargs(node_config.get("llm_configs", None)))
    llm_configs.update(llm_config or {})
    planner = create_planner(
        llm_configs,
        agent_id=agent_id,
        seed=seed,
        n_labs=n_labs,
        setting=space.llm_context().get("setting"),
        warmup_rounds=warmup_rounds,
    )

    return ADKOv2Agent(
        agent_id=agent_id,
        index=index,
        surrogate=surrogate,
        space=space,
        evaluator=evaluator,
        weights=create_weights(algorithm.get("weights", None)),
        q_bits=int(algorithm.get("q_bits", 8)),
        memory_rounds=int(algorithm.get("memory_rounds", 3)),
        degree=len(neighbors.recv_from),
        neighbor_indices=neighbor_indices,
        trust=bool(algorithm.get("trust", True)),
        trust_eta=float(algorithm.get("trust_eta", 0.1)),
        warmup_rounds=warmup_rounds,
        objective=str(algorithm.get("objective", "maximize")),
        seed=seed,
        planner=planner,
        n_rounds=int(node_config.num_epochs),
    )
