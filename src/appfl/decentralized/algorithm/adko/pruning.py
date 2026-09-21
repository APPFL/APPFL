"""Token pruning policies for ADKO memory budgets.

Algorithm 1 requires bounded token memory, ``|K_i^t| <= B``. Algorithm 2 prunes
tokens by keeping high-fidelity, high-confidence, recent evidence. This matters
because the Rillo et al. (ADKO) regret bound depends on preserving high average
fidelity ``eta_bar`` under the memory budget.
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
    """Keep tokens with the highest advantage ``c``."""

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
    """Keep a random subset of tokens.

    Args:
        seed: Seed combined with round and buffer size.
    """

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
