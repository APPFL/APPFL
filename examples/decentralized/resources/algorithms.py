"""Which decentralized algorithm a node config runs, and the little a launcher needs from it.

The launcher decides the transport; the config decides the experiment -- and the algorithm is
part of the experiment. A node config names it at the top level::

    algorithm: adko_v2          # adko (the default, v1) | adko_v2 (the settled v2 method)

This is the one place that maps the name to the package, so a launcher never imports an
algorithm directly and runs any of them unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from omegaconf import DictConfig

from llm_cli import llm_config_from_args

ALGORITHMS = ("adko", "adko_v2")


@dataclass
class Algorithm:
    """What a launcher needs: a meter, a way to build an agent, and a per-round hook."""

    name: str
    meter: Any
    _create: Callable[..., Any]
    end_round: Callable[[Sequence[Any]], None]

    def create_agent(
        self,
        node_config: DictConfig,
        neighbors: Any,
        args: argparse.Namespace,
        topology: Optional[Any] = None,
    ) -> Any:
        return self._create(node_config, neighbors, args, topology)

    def close(self, agents: Sequence[Any]) -> None:
        """Release what each agent holds open, for algorithms whose agents hold anything."""
        for agent in agents:
            close = getattr(agent, "close", None)
            if close is not None:
                close()


def load_algorithm(node_config: DictConfig) -> Algorithm:
    """The algorithm ``node_config`` names, with a fresh meter."""
    name = str(node_config.get("algorithm", "adko"))
    if name == "adko":
        from appfl.decentralized.algorithm.adko import create_agent
        from appfl.decentralized.algorithm.adko.metrics import ADKOMeter

        meter = ADKOMeter()

        def create(config, neighbors, args, topology):
            return create_agent(
                config,
                neighbors,
                meter,
                llm_config=llm_config_from_args(args),
                topology=topology,
            )

        return Algorithm(name, meter, create, meter.record_fidelity)

    if name == "adko_v2":
        from appfl.decentralized.algorithm.adko_v2 import ADKOv2Meter, create_agent

        meter = ADKOv2Meter()

        def create(config, neighbors, args, topology):
            if args.llm:
                raise SystemExit(
                    "--llm configures ADKO v1's language model. ADKO v2's planner (L3) is "
                    "set in the node config's `llm_configs` block; see "
                    "appfl.decentralized.algorithm.adko_v2.builder.create_planner."
                )
            return create_agent(config, neighbors, meter, topology=topology)

        return Algorithm(name, meter, create, meter.record_trust)

    raise ValueError(f"unknown algorithm {name!r}; available: {list(ALGORITHMS)}")
