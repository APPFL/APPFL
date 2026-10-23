"""Peer-token memory ``K_i``: arrival order, deduplication, FIFO with the best-so-far bit.

The settled method keeps a FIFO memory of **3 rounds** -- a budget of ``3 x degree`` tokens,
since each neighbour sends one a round -- and receivers keep **each peer's newest flagged
token past FIFO**, so a peer's best stays in view until it finds a better one. v1's
fidelity-aware pruning (Algorithm 2) is gone: the findings measured FIFO at least as good at
every budget, and with a running median old ``q`` values go stale anyway.

The owner never stores its own tokens -- its own results are already in its GP.

Ported from the reference implementation (FIFO policy only), so the order of tokens matches the
reference token for token: the peer field sums over memory in this order, and a different
summation order is a different last bit.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_v2.knowledge_token import (
    TAG_BEST,
    KnowledgeToken,
)


class TokenMemory:
    """Neighbours' tokens, in arrival order, pruned FIFO to ``budget``.

    :param budget: how many tokens to keep. The builder sets ``memory_rounds x degree``.
    :param kernel: ``"hamming"`` or ``"euclidean"``; read by the peer field.
    :param bandwidth: the kernel's ``sigma``.
    :param keep_tags: each source's newest token with one of these tags is pruned last.
    """

    def __init__(
        self,
        budget: int,
        kernel: str = "euclidean",
        bandwidth: float = 1.0,
        keep_tags: Sequence[int] = (TAG_BEST,),
    ):
        if budget is None or int(budget) < 0:
            raise ValueError("token budget must be non-negative")
        self.budget = int(budget)
        self.kernel, self.bandwidth = kernel, float(bandwidth)
        self.keep_tags = tuple(keep_tags)
        self.tokens: List[KnowledgeToken] = []

    def __len__(self) -> int:
        return len(self.tokens)

    def add(self, tokens: Iterable[KnowledgeToken]) -> None:
        """Append new tokens in the order given; a token already held is not added twice.

        Under a graph with cycles, or a transport that retries, the same token can arrive
        twice, and counting it twice would double its sender's pull.
        """
        for tok in tokens:
            if all(t.key != tok.key for t in self.tokens):
                self.tokens.append(tok)

    def newest(self, tags: Sequence[int]) -> Set[int]:
        """Indices of each source's newest token whose tag is in ``tags``."""
        out = {}
        for i, t in enumerate(self.tokens):
            if t.tag in tags and t.round >= out.get(t.source, (-1, 0))[0]:
                out[t.source] = (t.round, i)
        return {i for _, i in out.values()}

    def best(self) -> Set[int]:
        """Each source's newest flagged token: its best so far, as far as I know."""
        return self.newest((TAG_BEST,))

    def prune(self) -> List[KnowledgeToken]:
        """Drop the oldest tokens until the budget holds; returns the dropped ones.

        Oldest by round, then by arrival. A source's newest flagged token sorts after every
        unprotected one, so it goes only if protected tokens alone exceed the budget.
        """
        excess = len(self.tokens) - self.budget
        if excess <= 0:
            return []
        keep = self.newest(self.keep_tags)
        order = sorted(
            range(len(self.tokens)),
            key=lambda i: (i in keep, self.tokens[i].round, i),
        )
        drop = set(order[:excess])
        dropped = [t for i, t in enumerate(self.tokens) if i in drop]
        self.tokens = [t for i, t in enumerate(self.tokens) if i not in drop]
        return dropped

    # -- the arrays the peer field reads ----------------------------------------------

    @property
    def X(self) -> np.ndarray:
        return np.array([t.x for t in self.tokens], dtype=float)

    @property
    def q(self) -> np.ndarray:
        return np.array([t.q for t in self.tokens], dtype=float)

    @property
    def src(self) -> np.ndarray:
        return np.array([t.source for t in self.tokens], dtype=int)

    def counts(self) -> Counter:
        return Counter(t.source for t in self.tokens)

    def sources(self) -> Tuple[int, ...]:
        """Distinct senders in memory, in order of first appearance."""
        return tuple(dict.fromkeys(t.source for t in self.tokens))
