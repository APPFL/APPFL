"""``DeOptNodeAgent`` -- a decentralized optimization agent run as an APPFL DFL node.

The algorithm is not here, and nothing in this class is specific to one. ADKO is the
reference implementation -- :class:`~appfl.decentralized.algorithm.adko.ADKOAgent` follows
Algorithm 1 against :class:`~appfl.decentralized.protocol.AgentProtocol` and knows nothing
about APPFL: no agent base class, no communicator, no config loading. This class is the
adapter that lets any such agent be driven by an APPFL launcher, and it is deliberately
thin. A second decentralized optimizer with ``observe``/``act`` runs through it unchanged.

It is thin because a DFL round and an ADKO round are the same three steps under different
names::

    DFL      train()             get_parameters(r)        aggregate_parameters(peers)
    DeOpt    act(r) -> tokens    the tokens               observe(tokens)

So the adapter maps three methods and inherits everything else. What it inherits is the point:
round-tagged publication, the two-round retention, the blocking collect, the logger and the
configuration loading all come from :class:`~appfl.agent.DFLNodeAgent` and are identical to
what the MNIST example uses. The transport carries tokens exactly as it carries parameters,
because it never looks inside either.

The wrapped agent stays reachable as ``node.algorithm_agent``, and deliberately so: there is
no attribute forwarding here. ``node.space`` is an error, ``node.algorithm_agent.space`` is
the design space. The adapter owns the APPFL surface, the algorithm owns its own, and which
one you are talking to is visible at the call site rather than resolved at runtime.

Code that only wants the algorithm -- a reporting helper, an evaluation harness -- is best
handed ``node.algorithm_agent`` directly, which keeps it working unchanged against a plain
:class:`~appfl.decentralized.protocol.AgentProtocol`.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf

from appfl.agent.dfl_node import DFLNodeAgent
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.metrics import Meter
from appfl.decentralized.neighbor import Neighbors
from appfl.decentralized.protocol import AgentProtocol


class DeOptNodeAgent(DFLNodeAgent):
    """A DFL node whose payload is knowledge tokens rather than model parameters.

    :param algorithm_agent: the thing that actually runs rounds -- an
        :class:`~appfl.decentralized.protocol.AgentProtocol`. Typed against the protocol
        rather than a concrete class because nothing here is algorithm-specific.
    :param dfl_node_agent_config: the node's APPFL config. Synthesized from the algorithm
        agent's id when omitted, so an agent built in a test needs no config at all.
    :param budget: cap on what may go to one neighbor in one round. ``None`` is unlimited.
    :param meter: where traffic is counted. ``None`` counts nothing.
    """

    #: Columns of the per-round progress line, in order. Deliberately narrow enough to stay
    #: readable in a four-terminal gRPC run, where each process prints only its own rounds.
    _log_columns: Sequence[str] = (
        "Round",
        "Phase",
        "Best",
        "Tokens",
        "Emit",
        "Recv",
        "EtaBar",
        "PeerTerm",
    )

    def __init__(
        self,
        algorithm_agent: AgentProtocol,
        dfl_node_agent_config: Optional[DictConfig] = None,
        neighbors: Optional[Neighbors] = None,
        budget: Optional[CommBudget] = None,
        meter: Optional[Meter] = None,
        **kwargs,
    ) -> None:
        if dfl_node_agent_config is None:
            dfl_node_agent_config = OmegaConf.create(
                {"node_id": str(algorithm_agent.agent_id), "train_configs": {}}
            )
        self.algorithm_agent = algorithm_agent
        super().__init__(
            dfl_node_agent_config=dfl_node_agent_config, neighbors=neighbors, **kwargs
        )
        self.budget = budget
        self.meter = meter
        self._emitted: List[Any] = []
        self._logged_title = False

    # -- the three steps ----------------------------------------------------------------

    def train(self, **kwargs) -> None:
        """Run one round of the algorithm, then publish whatever it chose to broadcast.

        The round number is this node's next unpublished round, so the algorithm is told the
        same ``round_idx`` it would have been told by a driver that owned the loop.
        """
        self._emitted = list(self.algorithm_agent.act(self._round + 1))
        if self.meter is not None:
            self.meter.record_emitted(self._emitted)
        self.publish_parameters()

    def _payload_to_publish(self) -> List[Any]:
        """The tokens this round produced, trimmed to the budget and counted on the way out."""
        tokens = list(self._emitted)
        if self.budget is not None:
            admitted = self.budget.admits(tokens)
            if self.meter is not None:
                self.meter.tokens_dropped_budget += len(tokens) - len(admitted)
            tokens = admitted
        if self.meter is not None:
            for _ in self.neighbors.send_to:
                self.meter.record_sent(tokens)
        return tokens

    def aggregate_parameters(
        self, neighbor_models: Union[Dict[str, Any], Sequence[Any]], **kwargs
    ) -> None:
        """Hand the neighbors' tokens to the algorithm; it merges them at the top of `act`.

        Returns nothing, unlike the model-averaging case. There is no mixed model to hand
        back: what arrived is private state until the next round folds it into token memory.
        """
        payloads = (
            neighbor_models.values()
            if isinstance(neighbor_models, dict)
            else neighbor_models
        )
        tokens: List[Any] = []
        for payload in payloads:
            if payload:
                tokens.extend(payload)
        self.algorithm_agent.observe(tokens)
        self._log_round(len(tokens))

    def _log_round(self, num_received: int) -> None:
        """Log one line for the round that just finished.

        Here rather than in :meth:`train` because a round is not over until the neighbors'
        tokens are in. ``Recv`` is the only column not knowable earlier, and it is the one
        that matters most when something is wrong: a node that is optimizing happily but
        receiving nothing looks identical to a healthy one in every other column.

        Only the algorithm's *public* surface is read -- ``last_decision`` is documented as
        the hook for drivers that log traces -- and every field is probed with ``getattr``,
        so an :class:`~appfl.decentralized.protocol.AgentProtocol` that exposes none of it
        still logs its rounds, with ``-`` in the columns it cannot answer.
        """
        agent = self.algorithm_agent
        if not self._logged_title:
            self.logger.log_title(list(self._log_columns))
            self._logged_title = True

        best = agent.best_so_far() if hasattr(agent, "best_so_far") else None
        # `last_decision` is None during warmup, when no acquisition ran at all. An agent
        # that does not define it is a different kind of agent, not one in warmup, so the
        # two cases are distinguished rather than both printed as "warmup".
        has_phase = hasattr(agent, "last_decision")
        decision = getattr(agent, "last_decision", None)
        self.logger.log_content(
            [
                self._round,
                ("acquire" if decision else "warmup") if has_phase else "-",
                best[1] if best is not None else "-",
                len(getattr(agent, "token_memory", ())),
                len(self._emitted),
                num_received,
                agent.mean_token_fidelity()
                if hasattr(agent, "mean_token_fidelity")
                else "-",
                # lam*G - gamma*Lambda: the whole contribution the neighbourhood made to
                # this round's choice. One number for "is the exchange changing decisions".
                decision["score_lam_G"] - decision["score_gamma_Lambda"]
                if decision
                else "-",
            ]
        )

