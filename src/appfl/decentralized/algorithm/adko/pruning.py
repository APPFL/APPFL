"""Fidelity-aware token pruning -- ADKO Algorithm 2.

Token memory is bounded (Constraint 3.2: ``|K_i^t| <= B``), so something must be discarded.
*What* gets discarded is load-bearing: Proposition 4 shows this policy keeps average fidelity
``eta_bar -> 1``, which is precisely what removes the linear compression term from the regret
bound and buys sublinear convergence. Pruning is not housekeeping, it is part of the algorithm.

The variants below are the ablation arms from the paper's Section 6.1 -- ADKO beats both
FIFO and naive sharing, and reproducing that gap is the cheapest end-to-end check that an
APPFL-side port is faithful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken


class TokenPruner(ABC):
    """Reduce a token buffer to at most ``budget`` entries."""

    @abstractmethod
    def prune(
        self, tokens: Sequence[KnowledgeToken], budget: int, current_round: int
    ) -> List[KnowledgeToken]:
        """Return at most ``budget`` tokens to retain."""


class FidelityAwarePruner(TokenPruner):
    """ADKO Algorithm 2. Keeps the highest ``eta_hat * c * exp(-alpha_tau * age)``.

    Three factors, each doing distinct work:

    * ``eta_hat`` -- how much mutual information survived binary quantization. A token whose
      outcome sat on the contextual baseline is a coin flip; drop it first.
    * ``c`` -- the advantage score again, weighting decisive evidence over marginal.
    * ``exp(-alpha_tau * age)`` -- recency, so a stale map of a region the sender has since
      moved past does not crowd out current findings.

    The paper writes this as a while-loop dropping the argmin one at a time; sorting once is
    equivalent and O(n log n) rather than O(n * (n - B)).
    """

    def __init__(self, alpha_tau: float = 0.1):
        self.alpha_tau = alpha_tau

    def prune(
        self, tokens: Sequence[KnowledgeToken], budget: int, current_round: int
    ) -> List[KnowledgeToken]:
        if len(tokens) <= budget:
            return list(tokens)
        ranked = sorted(
            tokens,
            key=lambda k: k.pruning_score(current_round, self.alpha_tau),
            reverse=True,
        )
        return ranked[:budget]

    def mean_fidelity(self, tokens: Sequence[KnowledgeToken]) -> float:
        """``eta_bar``, the quantity Proposition 4 bounds. Log it every round.

        If this drifts down as the budget tightens, the compression term in the regret bound
        is not being controlled and sublinear convergence is not expected.
        """
        if not tokens:
            return 1.0
        return sum(k.fidelity() for k in tokens) / len(tokens)


class ConfidencePruner(TokenPruner):
    """Keep the highest-advantage tokens. The v2 many-task default.

    Recency breaks exact ties and nothing more, so this differs from :class:`FIFOPruner` in
    exactly one controlled choice -- which is what makes the pair a clean ablation.

    Note what is *absent* relative to :class:`FidelityAwarePruner`: no fidelity factor and no
    recency decay. The many-task tokens carry no fidelity at all, and the implementation
    prunes on confidence alone. The theory still defines fidelity (Definition 3) and still
    relies on ``eta_bar -> 1`` for the compression term (Proposition 4), so theory and
    implementation have diverged here; worth knowing before reading too much into either.
    """

    def prune(
        self, tokens: Sequence[KnowledgeToken], budget: int, current_round: int
    ) -> List[KnowledgeToken]:
        if len(tokens) <= budget:
            return list(tokens)
        ranked = sorted(
            tokens,
            key=lambda k: (k.advantage, k.provenance.round),
            reverse=True,
        )
        return ranked[:budget]


class RandomPruner(TokenPruner):
    """Uniformly random retention. The control arm: any selective policy must beat it."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def prune(
        self, tokens: Sequence[KnowledgeToken], budget: int, current_round: int
    ) -> List[KnowledgeToken]:
        import random

        if len(tokens) <= budget:
            return list(tokens)
        rng = random.Random((self.seed, current_round, len(tokens)).__hash__())
        return rng.sample(list(tokens), budget)


class FIFOPruner(TokenPruner):
    """Recency only -- the ``ADKO-FIFO`` ablation arm from Section 6.1."""

    def prune(
        self, tokens: Sequence[KnowledgeToken], budget: int, current_round: int
    ) -> List[KnowledgeToken]:
        if len(tokens) <= budget:
            return list(tokens)
        return sorted(tokens, key=lambda k: k.provenance.round, reverse=True)[:budget]


def merge(
    own: Sequence[KnowledgeToken], incoming: Sequence[KnowledgeToken]
) -> List[KnowledgeToken]:
    """ADKO Algorithm 1 step 1, ``MERGE(K_i^{t-1}, {k_j^{t-1}})``.

    Deduplicates on ``token_id``, since under a graph with cycles the same token can arrive
    by more than one path -- and double-counting a peer success would inflate the attraction
    term ``G_i`` in proportion to how well-connected the sender happens to be.
    """
    seen = set()
    merged: List[KnowledgeToken] = []
    for token in list(own) + list(incoming):
        if token.provenance.token_id in seen:
            continue
        seen.add(token.provenance.token_id)
        merged.append(token)
    return merged
