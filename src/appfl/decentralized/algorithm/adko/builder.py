"""Build an :class:`ADKOAgent` from configuration -- the ADKO half of config loading.

The generic half is :mod:`appfl.decentralized.config`, which reads the graph, the round count
and the communication budget. This module reads the keys only ADKO has: the weights of
Eq. (1), the contextual baseline, the pruning rule, and the surrogate/space/evaluator the
agent is built around.

It is the decentralized counterpart of :class:`appfl.agent.ClientAgent`, which does the same
job for federated learning -- take a config, resolve the paths in it, and hand back a
configured agent. Keeping it here rather than in ``examples/`` means every site in a
distributed run constructs its agent through the same code, which is what makes "the same
experiment three ways" checkable rather than merely intended.

Schema, under ``agent_configs.algorithm_configs`` in the federation config::

    weights:   {preset: many_task|suzuki, dim: N, <any ReasoningWeights field>}
    baseline:  {baseline: running_median|fixed|..., baseline_kwargs: {...}}
    pruner:    {pruner: fidelity|confidence|fifo|random, pruner_kwargs: {...}}
    token_budget, alpha_tau, warmup_rounds, objective, emit_insight, total_proposals,
    n_lm_candidates, n_local_candidates
"""

from __future__ import annotations

from typing import Callable

from omegaconf import DictConfig

from appfl.decentralized.config import as_kwargs, load_callable
from appfl.decentralized.metrics import Meter, metered_evaluator
from appfl.decentralized.neighbor import Neighbors
from appfl.decentralized.topology import Topology

from appfl.decentralized.algorithm.adko.agent import ADKOAgent
from appfl.decentralized.algorithm.adko.baseline import build_baseline
from appfl.decentralized.algorithm.adko.llm import LLMConfig, build_language_model
from appfl.decentralized.algorithm.adko.pruning import (
    ConfidencePruner,
    FIFOPruner,
    FidelityAwarePruner,
    RandomPruner,
)
from appfl.decentralized.algorithm.adko.reasoning import ReasoningWeights

__all__ = ["PRUNERS", "WEIGHT_PRESETS", "create_weights", "create_agent"]

PRUNERS: dict[str, Callable[..., object]] = {
    "fidelity": FidelityAwarePruner,
    "confidence": ConfidencePruner,
    "fifo": FIFOPruner,
    "random": RandomPruner,
}

WEIGHT_PRESETS: dict[str, Callable[..., ReasoningWeights]] = {
    "many_task": ReasoningWeights.many_task,
    "suzuki": ReasoningWeights.suzuki,
}


def create_weights(config: DictConfig | None) -> ReasoningWeights:
    """Eq. (1)'s weights: a published preset, then any explicit overrides on top.

    Presets exist because ``beta``, ``lam`` and ``gamma`` are not independent knobs -- the
    Suzuki study's (2, 4, 32) goes with Hamming distance, fidelity weighting and
    mixing-weight normalization, and mixing halves of two configurations is how a
    reproduction silently stops reproducing.
    """
    fields = as_kwargs(config)
    preset = fields.pop("preset", None)
    dim = fields.pop("dim", None)
    if preset is not None:
        if preset not in WEIGHT_PRESETS:
            raise ValueError(
                f"unknown weights preset {preset!r}; available: {sorted(WEIGHT_PRESETS)}"
            )
        weights = (
            WEIGHT_PRESETS[preset](dim=int(dim))
            if dim is not None
            else WEIGHT_PRESETS[preset]()
        )
    else:
        weights = ReasoningWeights()
    for key, value in fields.items():
        if value is None:
            continue
        if not hasattr(weights, key):
            raise ValueError(f"unknown ReasoningWeights field {key!r}")
        setattr(weights, key, value)
    return weights


def _mixing_weight(
    agent_id: str, neighbors: Neighbors, topology: Topology | None
) -> Callable[[str], float]:
    """``pi_ij`` as a function of the peer id, from whichever source this node has.

    A simulation holds the graph and asks it. A deployed site holds only its own neighbors,
    so the weight comes from the ``neighbors`` block -- named per peer when the operator sets
    them, uniform over the closed in-neighborhood otherwise, which is the same number the
    graph would have given. Non-neighbors get 0 either way, so a token arriving by an
    unexpected path is ignored rather than silently weighted.
    """
    if topology is not None:
        return lambda other, me=agent_id: topology.mixing_weight(me, other)

    uniform = 1.0 / (len(neighbors.recv_from) + 1)

    def weight(other: str) -> float:
        if other == agent_id:
            return neighbors.weights.get(other, uniform)
        if other not in neighbors.recv_from:
            return 0.0
        return neighbors.weights.get(other, uniform)

    return weight


def create_agent(
    node_config: DictConfig,
    neighbors: Neighbors,
    meter: Meter,
    llm_config: LLMConfig | None = None,
    topology: Topology | None = None,
) -> ADKOAgent:
    """Build one agent from one node's config: once per node, whatever the transport.

    Every launcher goes through here, so an agent is configured identically no matter which
    transport carries its tokens -- which is what makes cross-backend comparison mean
    anything.

    One config, complete on its own: the federation-wide file this used to be paired with is
    gone, so what was ``federation_config.agent_configs.algorithm_configs`` is now just
    ``node_config.algorithm_configs``. Resource paths are resolved as given, so they are
    relative to the working directory -- which every launcher expects to be ``examples/``.

    :param neighbors: who this node exchanges with. In a deployment it is all the node knows
        about the federation, which is why the mixing weight comes from here when no graph is
        available.
    :param topology: the whole graph, when a simulation launcher has one. Supplies the mixing
        weight and the node's index in the roster; both fall back to ``node_config`` without it.
    """
    algorithm = node_config.algorithm_configs

    agent_id = str(node_config.node_id)
    if topology is not None:
        index = list(topology.node_ids).index(agent_id)
    else:
        index = int(node_config.get("node_index", 0))
    seed = int(node_config.get("seed", 0)) + index

    space_configs = node_config.space_configs
    space = load_callable(space_configs.space_path, space_configs.space_name)(
        **as_kwargs(space_configs.get("space_kwargs", None))
    )

    surrogate_configs = node_config.surrogate_configs
    surrogate = load_callable(
        surrogate_configs.surrogate_path, surrogate_configs.surrogate_name
    )(**as_kwargs(surrogate_configs.get("surrogate_kwargs", None)))

    evaluator_configs = node_config.evaluator_configs
    evaluator = load_callable(
        evaluator_configs.evaluator_path, evaluator_configs.evaluator_name
    )(**as_kwargs(evaluator_configs.get("evaluator_kwargs", None)))
    # Priced as if it were a real experiment, so the meter's compute_seconds means something
    # even when the stand-in objective is instant.
    evaluator = metered_evaluator(
        evaluator, meter, cost_seconds=float(evaluator_configs.get("cost_seconds", 0.0))
    )

    baseline_configs = as_kwargs(algorithm.get("baseline", None))
    pruner_configs = as_kwargs(algorithm.get("pruner", None))
    pruner_name = str(pruner_configs.get("pruner", "fidelity"))
    if pruner_name not in PRUNERS:
        raise ValueError(
            f"unknown pruner {pruner_name!r}; available: {sorted(PRUNERS)}"
        )

    return ADKOAgent(
        agent_id=agent_id,
        surrogate=surrogate,
        space=space,
        evaluator=evaluator,
        # 1/(|N_i|+1), not Metropolis-Hastings: the uniform weight is what the reference
        # uses. `Topology.mixing_weight` is that same quantity on this branch, normalized
        # over the node's *incoming* set so the mixing matrix stays row-stochastic -- equal
        # for the undirected graphs ADKO assumes.
        mixing_weight=_mixing_weight(agent_id, neighbors, topology),
        baseline=build_baseline(
            str(baseline_configs.get("baseline", "running_median")),
            **as_kwargs(baseline_configs.get("baseline_kwargs", None)),
        ),
        # None unless an LLM is configured -- that is the LM-free ablation arm, which is how
        # the reference's non-LLM study runs.
        language_model=(
            build_language_model(llm_config, agent_id)
            if llm_config is not None and llm_config.enabled
            else None
        ),
        pruner=PRUNERS[pruner_name](
            **as_kwargs(pruner_configs.get("pruner_kwargs", None))
        ),
        weights=create_weights(algorithm.get("weights", None)),
        token_budget=int(algorithm.get("token_budget", 40)),
        n_lm_candidates=int(algorithm.get("n_lm_candidates", 10)),
        n_local_candidates=int(algorithm.get("n_local_candidates", 10)),
        alpha_tau=float(algorithm.get("alpha_tau", 0.01)),
        emit_insight=bool(
            algorithm.get("emit_insight", False)
            and llm_config is not None
            and llm_config.enabled
            and llm_config.emit_insight
        ),
        objective=str(algorithm.get("objective", "maximize")),
        warmup_rounds=int(algorithm.get("warmup_rounds", 5)),
        total_proposals=algorithm.get("total_proposals", None),
        seed=seed,
    )
