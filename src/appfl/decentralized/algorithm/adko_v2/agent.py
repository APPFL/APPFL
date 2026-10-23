"""``ADKOv2Agent`` -- the settled ADKO v2 method, against APPFL's transport and budgets.

One round, in the reference implementation's order, so the two can be diffed by eye and
compared bit for bit:

1. **Plan** (optional). After warmup, with probability ``alpha`` the L3 planner picks this
   round's experiment. Its pick is checked, and the score's pick stands if it fails.
2. **Choose.** Warmup rounds run the shared warmup points. After that, score every point of my
   grid with ``R = mu + 2 sigma + 2 G - 2 Lambda`` (``reasoning.py``) and take the best one I
   have not run.
3. **Run** the experiment -- the only place ground truth is touched.
4. **Learn** privately: refit the GP on all my results.
5. **Encode** one token: the shared input, ``q`` against my running median, and the
   best-so-far bit when this result beat everything I ran before (``knowledge_token.py``).
6. **Receive** (``observe``): update trust from each neighbour's new tokens, then add them to
   my FIFO memory and prune it to 3 rounds (``trust.py``, ``memory.py``).

The broadcast between 5 and 6 is the runtime's job -- this class never sees a transport.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.protocol import AgentProtocol
from appfl.decentralized.algorithm.adko_v2.components import (
    DesignSpace,
    Planner,
    Surrogate,
)
from appfl.decentralized.algorithm.adko_v2.knowledge_token import (
    KnowledgeToken,
    Quantizer,
    encode_token,
    score_against,
)
from appfl.decentralized.algorithm.adko_v2.memory import TokenMemory
from appfl.decentralized.algorithm.adko_v2.reasoning import (
    Calibrator,
    PeerField,
    ReasoningWeights,
    score,
)
from appfl.decentralized.algorithm.adko_v2.trust import (
    TrustState,
    agreement,
    relative_sigma,
)

#: The score's terms at a point, in the order the reference logs them.
TERMS = ("mu", "beta_sigma", "lam_G", "gamma_Lam")


def _rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


class ADKOv2Agent(AgentProtocol):
    """One ADKO v2 agent. Private surrogate, private data, token-only communication.

    :param index: this agent's position in the federation. It is the ``source`` its tokens
        carry, and it seeds its private streams together with ``seed``.
    :param evaluator: runs the experiment, ``point -> y``.
    :param memory_rounds: FIFO memory in rounds; the budget is ``memory_rounds x degree``.
    :param degree: how many neighbours send to this agent.
    :param neighbor_indices: their indices, when known (a simulation knows them; a deployed
        site learns them from the first tokens it receives).
    :param trust: odds-weighted trust (on in the settled method); off weighs peers equally.
    :param seed: the run seed. Every private stream is derived from ``(seed, index)``, as in
        the reference, so an agent draws the same numbers whichever process it runs in.
    :param planner: the optional L3 module; ``None`` is the LM-free method.
    :param n_rounds: the run's length, which the planner is told.
    """

    def __init__(
        self,
        agent_id: str,
        index: int,
        surrogate: Surrogate,
        space: DesignSpace,
        evaluator: Callable[[Any], float],
        weights: Optional[ReasoningWeights] = None,
        q_bits: int = 8,
        memory_rounds: int = 3,
        degree: int = 1,
        neighbor_indices: Sequence[int] = (),
        trust: bool = True,
        trust_eta: float = 0.1,
        warmup_rounds: int = 5,
        objective: str = "maximize",
        seed: int = 0,
        planner: Optional[Planner] = None,
        n_rounds: Optional[int] = None,
    ):
        if objective not in ("maximize", "minimize"):
            raise ValueError("objective must be 'maximize' or 'minimize'")
        if int(memory_rounds) <= 0:
            raise ValueError("memory_rounds must be positive")
        self.agent_id = str(agent_id)
        self.index = int(index)
        self.surrogate = surrogate
        self.space = space
        self.evaluator = evaluator
        self.weights = weights or ReasoningWeights()
        self.quantizer = Quantizer(q_bits)
        self.memory_rounds = int(memory_rounds)
        self.memory = TokenMemory(
            self.memory_rounds * max(int(degree), 1),
            self.weights.kernel,
            self.weights.bandwidth,
        )
        self.trust = TrustState(neighbor_indices, eta=trust_eta) if trust else None
        self.warmup_rounds = int(warmup_rounds)
        self.objective = objective
        self.seed = int(seed)
        self.planner = planner
        self.n_rounds = n_rounds

        # Private streams, keyed on (seed, index) exactly as the reference keys them.
        self.token_rng = np.random.default_rng([self.seed, self.index, 2026])
        self.rng = np.random.default_rng([self.seed, self.index, 2025])

        # The candidate grid, embedded once. ``_seen`` marks the rows already run.
        self._grid = list(space.enumerate())
        if not self._grid:
            raise ValueError(f"{type(space).__name__} enumerated no points")
        self._Z = np.asarray(space.grid(), dtype=float)
        if self._Z.shape[0] != len(self._grid):
            raise ValueError("space.grid() must have one row per enumerated point")
        self._row = {self._key(z): r for r, z in enumerate(self._Z)}
        self._seen = np.zeros(len(self._grid), dtype=bool)
        warm = space.warmup_points(self.seed, self.warmup_rounds)
        self.warmup_points = list(warm) if warm is not None else None

        # Private state. None of it is ever transmitted (Constraint 3.1).
        self._points: List[Any] = []
        self._y: List[float] = []  # outcomes as measured
        self._u: List[
            float
        ] = []  # outcomes on the maximized scale, what is learned from
        self._round = -1
        self.sigma_ref: Optional[float] = None
        self.terms: Optional[Tuple[np.ndarray, ...]] = None

        #: Run traces, read by drivers and reports, never by peers.
        self.steps: List[Dict[str, Any]] = []
        self.calibration: List[Dict[str, Any]] = []
        self.sent: List[KnowledgeToken] = []
        self.trust_trace: List[Tuple[int, Dict[int, float]]] = []
        #: Score terms at the last acquisition round's pick; ``None`` in warmup.
        self.last_decision: Optional[Dict[str, Any]] = None

    # -- AgentProtocol -----------------------------------------------------------------

    def act(self, round_idx: int) -> List[KnowledgeToken]:
        """One round: plan, choose, run, learn, encode. Returns the one token to send."""
        self._round = r = int(round_idx)

        # 1. The planner decides from the state at the top of the round, before choosing.
        decision = None
        if (
            self.planner is not None
            and r >= self.warmup_rounds
            and self.planner.wants(r, self.index)
        ):
            decision = self.planner.decide([self.context(r)])[0]

        # 2. Choose.
        if r < self.warmup_rounds:
            point = self._warmup_point(r)
            record: Dict[str, Any] = {
                "chose_by": "shared_warmup"
                if self.warmup_points is not None
                else "warmup"
            }
            self.last_decision = None
        else:
            point, diag = self._propose()
            self.calibration.append(dict(round=r, agent=self.index, **diag))
            record = {"chose_by": "acquisition"}
            if decision is not None:
                record["call_id"] = decision.get("call_id")
                reason = self.check(decision.get("x"))
                if reason is None:
                    record.update(
                        chose_by=f"llm_{decision.get('source', 'proposal')}",
                        acq_pick=self._as_list(point),
                        insight=decision.get("insight"),
                    )
                    point = self._grid[self._row_of(decision["x"])]
                else:
                    record["llm_rejected"] = reason
            record.update(self.components(point))
            self.last_decision = {
                "round": r,
                "chose_by": record["chose_by"],
                "score_mu": record["mu"],
                "score_beta_sigma": record["beta_sigma"],
                "score_lam_G": record["lam_G"],
                "score_gamma_Lambda": record["gamma_Lam"],
                "n_tokens_in_memory": len(self.memory),
            }

        # 3. Run. 4-5. Learn and encode.
        y = float(self.evaluator(point))
        token = self._learn(point, y, r, record.pop("insight", None))
        if self.planner is not None and record.get("call_id"):
            self.planner.record_outcome(record["call_id"], y, token.q)

        self.steps.append(
            dict(
                round=r,
                agent=self.index,
                theta=self._as_list(point),
                y=y,
                best_so_far=float(self.best_so_far()[1]),
                n_tokens=len(self.memory),
                **record,
            )
        )
        self.sent.append(token)
        return [token]

    def observe(self, tokens: Sequence[KnowledgeToken]) -> None:
        """6. Receive: trust learns from the new tokens, then they enter memory.

        Tokens are taken in (round, sender) order whatever order the transport delivered them
        in, which is the order the reference's synchronous loop produces -- so memory, and the
        sums over it, are identical under serial, MPI and gRPC.
        """
        peers = sorted(
            (t for t in tokens if t.source != self.index),
            key=lambda t: (t.round, t.source),
        )
        if self.trust is not None:
            for t in peers:
                self.trust.ensure(t.source)
            learn = self._round >= self.warmup_rounds and self.surrogate.fitted
            if learn and peers:
                self._update_trust(peers)
        self.memory.add(peers)
        self.memory.prune()
        if self.trust is not None:
            self.trust_trace.append((self._round, self.trust.snapshot()))

    # -- the steps -----------------------------------------------------------------------

    def source_weights(self) -> Optional[Dict[int, float]]:
        """How peers are mixed: ``None`` (equally) or trust's odds."""
        return None if self.trust is None else self.trust.weights()

    def _propose(self) -> Tuple[Any, Dict[str, Any]]:
        """Score the grid; return the best unobserved point and the round's calibration."""
        w = self.weights
        mu, sigma = self.surrogate.predict(self._Z)
        field = PeerField(self.memory, self.source_weights())
        G0, P0 = field(self._Z)
        calibrator = Calibrator(sigma, G0, P0, w.eps)
        G, L = calibrator.apply(G0, P0)
        self.terms = (mu, w.beta * sigma, w.lam * G, w.gamma * L)
        self.sigma_ref = float(sigma.max())
        R = score(mu, sigma, G, L, w.beta, w.lam, w.gamma)
        R[self._seen] = -np.inf
        k = int(np.argmax(R))
        diag = dict(calibrator.diagnostics)
        diag.update({f"rms_{t}": _rms(v) for t, v in zip(TERMS[1:], self.terms[1:])})
        return self._grid[k], diag

    def _learn(
        self, point: Any, y: float, round_idx: int, insight: Optional[str]
    ) -> KnowledgeToken:
        """Record the result, refit, and encode the token that goes out."""
        embedding = self.space.embed(point)
        u = y if self.objective == "maximize" else -y
        self._points.append(point)
        self._y.append(y)
        self._u.append(u)
        row = self._row.get(self._key(embedding))
        if row is not None:
            self._seen[row] = True
        self.surrogate.update(embedding, u)
        shared = self.space.privatize(embedding, self.token_rng)
        new_best = len(self._u) == 1 or u > max(self._u[:-1])
        return encode_token(
            agent_id=self.agent_id,
            source=self.index,
            round=round_idx,
            x_shared=shared,
            y=u,
            history=self._u,
            quantizer=self.quantizer,
            insight=insight,
            new_best=new_best,
        )

    def _update_trust(self, tokens: Sequence[KnowledgeToken]) -> None:
        """Agreement of each new token with my GP; one update per sender."""
        mu, sd = self.surrogate.predict(np.asarray([t.x for t in tokens], dtype=float))
        q_hat = [
            score_against(m, self._u, self.quantizer)
            for m in self.surrogate.raw_mean(mu)
        ]
        tilde = relative_sigma(sd, self.sigma_ref)
        by_source: Dict[int, List[float]] = {}
        for t, qh, st in zip(tokens, q_hat, tilde):
            by_source.setdefault(t.source, []).append(agreement(t.q, qh, st))
        for j, a in by_source.items():
            self.trust.update(j, a)

    # -- helpers -----------------------------------------------------------------------

    @staticmethod
    def _key(embedding: Sequence[float]) -> tuple:
        """Identity of a probe. Rounded so float noise cannot make one point look like two."""
        return tuple(round(float(v), 9) for v in embedding)

    def _row_of(self, point: Any) -> Optional[int]:
        """The grid row of a point, or ``None`` if it is not in my slice."""
        try:
            return self._row.get(self._key(self.space.embed(point)))
        except Exception:  # noqa: BLE001 - a malformed point is simply not in the grid
            return None

    @staticmethod
    def _as_list(point: Any) -> List[Any]:
        return list(point) if isinstance(point, (list, tuple, np.ndarray)) else [point]

    def _warmup_point(self, round_idx: int) -> Any:
        if self.warmup_points is not None and round_idx < len(self.warmup_points):
            return self.warmup_points[round_idx]
        free = np.flatnonzero(~self._seen)
        return self._grid[int(self.rng.choice(free))]

    def check(self, x: Any) -> Optional[str]:
        """``None`` if ``x`` is in my slice and untried; otherwise why not."""
        if x is None:
            return "no_proposal"
        row = self._row_of(x)
        if row is None:
            return "infeasible"
        if self._seen[row]:
            return "observed"
        return None

    def components(self, point: Any) -> Dict[str, float]:
        """The score's terms at a grid point, from the last :meth:`_propose`."""
        row = self._row_of(point)
        return {k: float(v[row]) for k, v in zip(TERMS, self.terms)}

    def context(self, round_idx: int) -> Dict[str, Any]:
        """What a planner may see: my own data, my allowed options, my memory."""
        ctx = {
            "agent": self.index,
            "setting": None,
            "round": round_idx,
            "n_rounds": self.n_rounds,
            "allowed": self.space.allowed(),
            "history": [
                {"round": k, "x": self._as_list(p), "y": y}
                for k, (p, y) in enumerate(zip(self._points, self._y))
            ],
            "memory": [
                {
                    "id": t.id,
                    "source": t.source,
                    "round": t.round,
                    "x": [int(v) for v in t.x],
                    "q": t.q,
                    "insight": t.insight,
                    "tag": t.tag,
                }
                for t in reversed(self.memory.tokens)
            ],
            "observed": [self._as_list(p) for p in self._points],
        }
        ctx.update(self.space.llm_context())
        return ctx

    # -- inspection --------------------------------------------------------------------

    @property
    def token_memory(self) -> List[KnowledgeToken]:
        return self.memory.tokens

    def best_so_far(self) -> Optional[Tuple[Any, float]]:
        if not self._y:
            return None
        k = int(np.argmax(self._u))
        return self._points[k], self._y[k]

    def mean_trust(self) -> Optional[float]:
        """Mean trust over neighbours; ``None`` without trust."""
        return None if self.trust is None else self.trust.mean()

    def describe(self) -> str:
        w = self.weights
        return (
            f"beta={w.beta} lam={w.lam} gamma={w.gamma} kernel={w.kernel} "
            f"sigma_s={w.bandwidth} calibration=match_rms eps={w.eps} "
            f"q_bits={self.quantizer.bits} memory={self.memory_rounds} rounds "
            f"({self.memory.budget} tokens) best_bit=on "
            f"trust={'odds' if self.trust is not None else 'off'}"
        )

    def close(self) -> None:
        """Release what the agent holds open: the planner's call log, cache and client."""
        if self.planner is not None:
            self.planner.close()

    def trace(self) -> Dict[str, Any]:
        """Everything this agent recorded, as plain data a driver can gather and write out."""
        return {
            "agent": self.index,
            "agent_id": self.agent_id,
            "steps": [dict(s) for s in self.steps],
            "calibration": [dict(c) for c in self.calibration],
            "tokens": [t.to_dict() for t in self.sent],
            "trust": [(r, dict(snapshot)) for r, snapshot in self.trust_trace],
            "llm": None if self.planner is None else dict(self.planner.totals),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "index": self.index,
            "points": list(self._points),
            "observations": list(self._y),
            "token_memory": [t.to_dict() for t in self.memory.tokens],
            "trust": None if self.trust is None else self.trust.snapshot(),
        }
