"""``ADKOAgent`` -- ADKO Algorithm 1, implemented against APPFL's transport and budgets.

This is the reference agent: ISU / NYU supply a :class:`Surrogate`, a :class:`LanguageModel`,
and a :class:`DesignSpace`; everything else here follows the paper step for step. The step
numbers in :meth:`act` map onto Algorithm 1 lines 1-12 so the two can be diffed by eye.

The one deliberate departure: evaluation (step 9) goes through an injected ``evaluator``
callable rather than being inlined. That seam is where DAISY Task 2.2 lives -- wrap the
evaluator in a budget-metering, deduplicating, fallback-substituting decorator and the agent
becomes compute-aware without a line of its own changing. See ``budget.py``.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from appfl.decentralized.protocol import AgentProtocol
from appfl.decentralized.algorithm.adko.baseline import BaseBaseline
from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken, encode_token
from appfl.decentralized.algorithm.adko.components import DesignSpace, LanguageModel, Surrogate
from appfl.decentralized.algorithm.adko.pruning import FidelityAwarePruner, TokenPruner, merge
from appfl.decentralized.algorithm.adko.reasoning import ReasoningWeights, score_candidates


class ADKOAgent(AgentProtocol):
    """One ADKO agent. Private surrogate, private data, token-only communication.

    :param evaluator: runs the actual experiment or simulation, ``phi_point -> y``. 
    :param mixing_weight: ``agent_id -> pi_ij``. To reproduce the paper, bind
        ``topology.uniform_weight`` -- that is what the reference uses. Returns 0 for
        non-neighbors, so tokens arriving by an unexpected path are ignored rather than
        silently over-weighted.
    :param baseline: supplies ``(b, scale)`` for token encoding. Use
        :class:`~appfl.decentralized.algorithm.adko.baseline.FixedBaseline` when "success" has a domain meaning
        (Suzuki: ``b = 50`` on a 0-100 yield scale) and
        :class:`~appfl.decentralized.algorithm.adko.baseline.RunningMedianBaseline` otherwise -- the latter is the
        paper's v2 recommended default, since it adapts to each agent's own objective scale.
    :param objective: ``"maximize"`` or ``"minimize"``; flips the sign convention in encoding.
    :param token_budget: ADKO's ``B``, the token-memory bound from Constraint 3.2.
    """

    def __init__(
        self,
        agent_id: str,
        surrogate: Surrogate,
        space: DesignSpace,
        evaluator: Callable[[Any], float],
        mixing_weight: Callable[[str], float],
        baseline: BaseBaseline,
        language_model: Optional[LanguageModel] = None,
        pruner: Optional[TokenPruner] = None,
        weights: Optional[ReasoningWeights] = None,
        token_budget: int = 40,
        n_lm_candidates: int = 10,
        n_local_candidates: int = 10,
        alpha_tau: float = 0.01,
        emit_insight: bool = True,
        objective: str = "maximize",
        warmup_rounds: int = 5,
        total_proposals: Optional[int] = None,
        seed: int = 0,
    ):
        self.agent_id = agent_id
        self.surrogate = surrogate
        self.space = space
        self.evaluator = evaluator
        self.mixing_weight = mixing_weight
        self.baseline = baseline
        self.objective = objective
        self.language_model = language_model
        self.pruner = pruner or FidelityAwarePruner(alpha_tau=alpha_tau)
        self.weights = weights or ReasoningWeights()
        self.token_budget = token_budget
        self.n_lm_candidates = n_lm_candidates
        self.n_local_candidates = n_local_candidates
        self.emit_insight = emit_insight
        self.warmup_rounds = warmup_rounds
        self.total_proposals = total_proposals
        self.rng = random.Random(seed)

        # Private state. None of this is ever transmitted (Constraint 3.1).
        self.token_memory: List[KnowledgeToken] = []
        self._observations: List[float] = []
        self._points: List[Any] = []
        self._inbox: List[KnowledgeToken] = []
        # Points already evaluated, keyed by their embedding, so the candidate pool can
        # exclude them. The reference does this by subtracting its observed set from the
        # full grid (`run_suzuki.py::_unobserved_candidates`); re-proposing a point that has
        # already been measured wastes a round of an expensive budget.
        self._observed_keys: set = set()
        self._best_round: int = -1

    # -- AgentProtocol -----------------------------------------------------------------

    def observe(self, tokens: Sequence[KnowledgeToken]) -> None:
        """Stage incoming peer tokens; they are merged at the top of the next round."""
        self._inbox.extend(tokens)

    def act(self, round_idx: int) -> List[KnowledgeToken]:
        """One full ADKO round. Returns the token to broadcast (zero or one, per 3.2)."""
        # Step 1: token aggregation.
        self.token_memory = merge(self.token_memory, self._inbox)
        self._inbox = []

        # Step 2: fidelity-aware pruning to the memory budget.
        self.token_memory = self.pruner.prune(
            self.token_memory, self.token_budget, round_idx
        )

        # Steps 3-4: candidate generation, then 5-8: score and select.
        #
        # Warmup short-circuits both. The reference proposes uniformly at random until it has
        # `WARMUP_ROUNDS` rounds and at least two observations
        # (`run_suzuki.py:1146`), because a GP fitted on nothing gives a flat posterior and
        # argmax over a flat posterior is not exploration -- it just returns whichever
        # candidate the sampler happened to emit first. Random gives real coverage.
        if self._in_warmup(round_idx):
            chosen = self._propose_random()
            if chosen is None:
                return []
            chosen_embedding = self.space.embed(chosen)
        else:
            candidates = self._generate_candidates()
            if not candidates:
                return []
            embeddings = [self.space.embed(c) for c in candidates]

            # Private (mu, sigma) from the surrogate; peer (G, Lambda) from token memory.
            # The two streams never mix before this point.
            posteriors = self.surrogate.posterior(embeddings)
            scores = score_candidates(
                embeddings,
                posteriors,
                self.token_memory,
                self.mixing_weight,
                self.weights,
            )
            best_idx = max(scores, key=lambda i: scores[i])
            chosen, chosen_embedding = candidates[best_idx], embeddings[best_idx]

        # Step 9: execution. The only place the agent touches ground truth.
        observation = self.evaluator(chosen)

        # Step 10: token encoding. `observation` dies inside encode_token -- from here on,
        # only the sign and the normalized magnitude exist.
        #
        # The observation is recorded *before* the baseline is read, so under a running
        # baseline it participates in its own median. That is what the reference does
        # (`_evaluate_and_append` appends to `y_s`, then `encode` takes `median(y_s)`), and
        # the ordering is worth preserving: reading the baseline first would make an agent's
        # very first token compare against an empty history.
        self.baseline.observe(observation)
        threshold, scale = self.baseline.current()
        insight = None
        if self.emit_insight and self.language_model is not None:
            insight = self.language_model.encode_insight(
                chosen_embedding, observation, threshold
            )
        token = encode_token(
            agent_id=self.agent_id,
            round=round_idx,
            observation=observation,
            threshold=threshold,
            scale=scale,
            objective=self.objective,
            embedding=chosen_embedding,
            space_id=getattr(self.space, "space_id", ""),
            insight=insight,
            insight_model=(
                getattr(self.language_model, "name", None) if insight else None
            ),
        )

        # Step 12: private GP update. (Step 11, broadcast, is the runtime's job -- the agent
        # does not know or care what transport carries the token.)
        self.surrogate.update(chosen_embedding, observation)
        if not self._observations or observation > max(self._observations):
            self._best_round = round_idx
        self._observations.append(observation)
        self._points.append(chosen)
        self._observed_keys.add(self._key(chosen_embedding))
        self.token_memory.append(token)

        return [token]

    # -- helpers -----------------------------------------------------------------------

    def _in_warmup(self, round_idx: int) -> bool:
        """``run_suzuki.py:1146`` -- random until enough rounds *and* enough observations."""
        return round_idx < self.warmup_rounds or len(self._observations) < 2

    def _key(self, embedding: Sequence[float]) -> tuple:
        """Identity of a probe for the observed-set filter. Rounded so floating-point noise
        does not make the same categorical point look like two."""
        return tuple(round(float(v), 9) for v in embedding)

    def _unobserved(self) -> Optional[List[Any]]:
        """This agent's allowed points minus the ones it has already measured.

        ``None`` when the space cannot be enumerated -- a continuous space -- in which case
        the caller samples instead. That fallback is an approximation of the reference, not
        the reference: it scores the whole unobserved grid every round.
        """
        everything = self.space.enumerate()
        if everything is None:
            return None
        return [
            point
            for point in everything
            if self._key(self.space.embed(point)) not in self._observed_keys
        ]

    def _propose_random(self) -> Optional[Any]:
        """Warmup, and the fallback when the candidate pool collapses."""
        pool = self._unobserved()
        if pool is not None:
            return self.rng.choice(pool) if pool else None
        sampled = self.space.sample(1)
        return sampled[0] if sampled else None

    def _progress(self) -> Dict[str, Any]:
        """What the reference's propose prompt calls the progress block.

        Lets the model tell "this region is promising and unexplored" from "this region is
        promising and I have already mined it", which peer tokens alone cannot express.
        """
        best = max(self._observations) if self._observations else None
        return {
            "n_obs": len(self._observations),
            "best_y": best,
            "rounds_since_improve": (
                0 if self._best_round < 0 else len(self._observations) - 1 - self._best_round
            ),
            "recent_improvement": (
                0.0
                if len(self._observations) < 2
                else max(self._observations[-5:]) - max(self._observations[:-5] or [best])
            ),
        }

    def _generate_candidates(self) -> List[Any]:
        """Steps 3-4. Mirrors ``run_suzuki.py::propose``'s pool construction.

        Three cases, in the reference's own order:

        * no LM, or ``total_proposals`` unset -- score the **entire unobserved set**. This is
          what the reference does by default, and it is why its results are not a function of
          how many candidates a sampler happened to draw.
        * LM candidates plus ``total_proposals = N`` -- take the model's picks, then fill to
          ``N`` with a uniform random draw from the remaining unobserved points, excluding
          the model's own picks so the pool holds no duplicates.
        * space not enumerable -- fall back to sampling around the incumbent. An
          approximation, flagged as such, for continuous spaces the reference never runs on.
        """
        unobserved = self._unobserved()

        lm_picks: List[Any] = []
        if self.language_model is not None:
            proposed = self.language_model.propose(
                self.token_memory,
                self.space,
                self.n_lm_candidates,
                history=list(zip(self._points, self._observations)),
                progress=self._progress(),
            )
            # Feasible, unobserved, unique -- the reference sanitises the model's output the
            # same way rather than trusting it (`_sanitise_llm_candidates`).
            seen = set()
            for point in proposed:
                key = self._key(self.space.embed(point))
                if key in self._observed_keys or key in seen:
                    continue
                seen.add(key)
                lm_picks.append(point)

        if unobserved is None:
            # Continuous fallback: perturb around the incumbent, plus the model's picks.
            candidates = list(lm_picks)
            if self._points:
                best_idx = max(
                    range(len(self._observations)), key=lambda i: self._observations[i]
                )
                candidates.extend(
                    self.space.local_perturbations(
                        self._points[best_idx], self.n_local_candidates
                    )
                )
            else:
                candidates.extend(self.space.sample(self.n_local_candidates))
            return candidates

        if self.total_proposals is None:
            # Score everything unobserved; the model's picks are already in there.
            return unobserved

        n_random = max(self.total_proposals - len(lm_picks), 0)
        lm_keys = {self._key(self.space.embed(p)) for p in lm_picks}
        pool = [p for p in unobserved if self._key(self.space.embed(p)) not in lm_keys]
        if len(pool) > n_random:
            pool = self.rng.sample(pool, n_random)
        return lm_picks + pool

    # -- inspection --------------------------------------------------------------------

    def best_so_far(self) -> Optional[Tuple[Any, float]]:
        if not self._observations:
            return None
        best_idx = max(
            range(len(self._observations)), key=lambda i: self._observations[i]
        )
        return self._points[best_idx], self._observations[best_idx]

    def mean_token_fidelity(self) -> float:
        """``eta_bar`` over current memory -- Proposition 4's quantity. Log it every round."""
        if not self.token_memory:
            return 1.0
        return sum(k.fidelity() for k in self.token_memory) / len(self.token_memory)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "observations": list(self._observations),
            "points": list(self._points),
            "token_memory": [k.to_dict() for k in self.token_memory],
        }
