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

from typing import Callable, Dict, Optional

from omegaconf import DictConfig

from appfl.decentralized.config import agent_ids, as_kwargs, load_callable
from appfl.decentralized.metrics import Meter, metered_evaluator
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

PRUNERS: Dict[str, Callable[..., object]] = {
    "fidelity": FidelityAwarePruner,
    "confidence": ConfidencePruner,
    "fifo": FIFOPruner,
    "random": RandomPruner,
}

WEIGHT_PRESETS: Dict[str, Callable[..., ReasoningWeights]] = {
    "many_task": ReasoningWeights.many_task,
    "suzuki": ReasoningWeights.suzuki,
}


def create_weights(config: Optional[DictConfig]) -> ReasoningWeights:
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


def create_agent(
    agent_config: DictConfig,
    federation_config: DictConfig,
    topology: Topology,
    meter: Meter,
    llm_config: Optional[LLMConfig] = None,
) -> ADKOAgent:
    """Build one agent: once per agent in a serial run, once per process everywhere else.

    Every launcher goes through here, so an agent is configured identically no matter which
    transport carries its tokens -- which is what makes cross-backend comparison mean
    anything.

    Resource paths in the two configs are expected to be absolute already;
    :func:`appfl.decentralized.config.load_federation_config` and
    :func:`~appfl.decentralized.config.load_agent_configs` resolve them at load time.
    """
    federation = federation_config.federation_configs
    shared = federation_config.agent_configs
    algorithm = shared.algorithm_configs

    agent_id = str(agent_config.agent_id)
    index = agent_ids(federation_config).index(agent_id)
    seed = int(federation.get("seed", 0)) + index

    space_configs = agent_config.space_configs
    space = load_callable(space_configs.space_path, space_configs.space_name)(
        **as_kwargs(space_configs.get("space_kwargs", None))
    )

    surrogate_configs = shared.surrogate_configs
    surrogate = load_callable(
        surrogate_configs.surrogate_path, surrogate_configs.surrogate_name
    )(**as_kwargs(surrogate_configs.get("surrogate_kwargs", None)))

    evaluator_configs = agent_config.evaluator_configs
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
        raise ValueError(f"unknown pruner {pruner_name!r}; available: {sorted(PRUNERS)}")

    return ADKOAgent(
        agent_id=agent_id,
        surrogate=surrogate,
        space=space,
        evaluator=evaluator,
        # uniform_weight, not Metropolis-Hastings: 1/(|N_i|+1) is what the reference uses.
        mixing_weight=lambda other, me=agent_id: topology.uniform_weight(me, other),
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
        pruner=PRUNERS[pruner_name](**as_kwargs(pruner_configs.get("pruner_kwargs", None))),
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
