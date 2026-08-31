"""The ADKO reasoning score -- Eq. (1), the thing each agent argmaxes to pick its next probe.

    R_i(theta) = U_i(theta)  +  beta * sigma_i(theta)  +  lambda * G_i(theta)  -  gamma * Lambda_i(theta)
                 \\_________/     \\________________/       \\______________/         \\__________________/
                  what I expect   how unsure I am          do my neighbours         do my neighbours
                  from my data    personally               succeed here?            fail here?

The first two terms are exactly GP-UCB over the agent's *private* posterior. The last two are
the collaboration, and they are computed entirely from peer tokens -- never from peer data.

Set ``lam = gamma = 0`` and it degenerates to independent per-agent GP-UCB, which is the
paper's "communication is necessary" lower bound and the natural no-communication baseline
for the DAISY AI-advantage comparison. (The reference implementation does exactly this: its
``INDEP`` arm reuses the ADKO loop with token broadcast disabled.)

Eq. (1) as typeset does not pin down three things, and the two published implementations
resolve them **differently**. They are therefore settings on :class:`ReasoningWeights`, with
:meth:`ReasoningWeights.suzuki` and :meth:`ReasoningWeights.many_task` as the two known-good
presets:

1. **Token weighting** -- ``c_k * eta_k`` (Suzuki: fidelity discounts the contribution as
   well as driving pruning) or ``c_k`` alone (many-task: its tokens carry no fidelity at all).
2. **Per-source normalization** -- each source's contribution is scaled to [0, 1] by its own
   weight sum either way, so a neighbor gets one equal-strength voice regardless of how many
   of its tokens survived pruning. What differs is the outer factor: the graph mixing weight
   ``pi_ij`` (Suzuki) or a plain average over sources present in memory (many-task).
3. **Kernel denominator** -- ``2 * sigma_s^2`` (Suzuki) or ``sigma_s^2`` (many-task, and the
   paper as written).

None of these is a detail at realistic bandwidths, and picking the wrong combination produces
a run that completes and quietly behaves like the no-communication arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken, Signal


def distance(
    a: Sequence[float], b: Sequence[float], metric: str = "euclidean"
) -> float:
    """Distance between two design-point embeddings.

    ``euclidean`` for continuous spaces. ``hamming`` -- the fraction of positions that
    differ -- for categorical spaces, which is what the Suzuki study uses: its design points
    are integer category indices (ligand, solvent, base, coupling partner), where numeric
    distance between category ids is meaningless.
    """
    if metric == "hamming":
        if not a:
            return 0.0
        return sum(1.0 for x, y in zip(a, b) if x != y) / len(a)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def bandwidth_for_dimension(dim: int, target_similarity: float = 0.2) -> float:
    """``sigma_s = sqrt((d / 6) / -log(s*))`` -- the paper's v2 bandwidth heuristic.

    In a normalized d-dimensional space two typical points sit at squared distance about
    ``d / 6``. Choosing the similarity ``s*`` you want peers to still have at that distance
    pins the bandwidth. Without this, peer influence vanishes as dimension grows and the
    social terms quietly stop doing anything -- the failure mode is silent, since the run
    still completes and simply behaves like the no-communication arm.

    ``d = 10, s* = 0.2`` gives ``sigma_s ~= 1.02``, the value the many-task study uses.
    """
    if not 0.0 < target_similarity < 1.0:
        raise ValueError("target_similarity must be in (0, 1)")
    return math.sqrt((dim / 6.0) / -math.log(target_similarity))


def similarity(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
    sigma_s: float = 1.0,
    metric: str = "euclidean",
    kernel: str = "sigma_sq",
) -> float:
    """``S(theta, theta_k) = exp(-d(phi(theta), phi(theta_k))^2 / denom)``.

    Operates on embeddings, never raw design points -- which is what lets peer influence be
    computed without violating Constraint 3.1.

    ``kernel`` picks the denominator, and the two published studies differ:
    ``"sigma_sq"`` gives ``sigma_s^2`` (the paper as written, and the many-task
    implementation) while ``"two_sigma_sq"`` gives ``2 * sigma_s^2`` (the Suzuki
    implementation's Gaussian form). A factor of two in the exponent is not cosmetic at these
    bandwidths, so it is a setting rather than a convention.

    ``sigma_s`` sets how far a peer's evidence reaches; under the Hamming metric it reads as
    "how many disagreeing positions still count as nearby". See
    :func:`bandwidth_for_dimension` for a principled starting value.
    """
    if not embedding_a or not embedding_b:
        return 0.0
    d = distance(embedding_a, embedding_b, metric)
    bw = max(sigma_s, 1e-12)
    denom = bw**2 if kernel == "sigma_sq" else 2 * bw**2
    return math.exp(-(d**2) / denom)


@dataclass
class ReasoningWeights:
    """``beta``, ``lambda``, ``gamma``, ``sigma_s`` from Eq. (1), plus the form choices.

    Defaults follow the paper's v2 "low-calibration starting point": ``beta = 2`` (a standard
    GP-UCB choice) and ``lam = gamma = 2``, giving peer successes and failures **symmetric**
    weight. That is a change from the Suzuki study's tuned ``(4, 32)``, where failures weighed
    eight times successes -- see :meth:`suzuki`.

    ``weight_by_fidelity`` and ``peer_normalization`` are where the two published
    implementations genuinely disagree, so they are settings rather than hardcoded:

    * the Suzuki implementation weights each token by ``c * eta`` and scales each source by
      the graph mixing weight ``pi_ij``;
    * the many-task implementation weights by ``c`` alone -- its tokens carry no fidelity at
      all -- and averages over the sources actually present in memory.
    """

    beta: float = 2.0
    lam: float = 2.0
    gamma: float = 2.0
    sigma_s: float = 1.0
    metric: str = "euclidean"
    kernel: str = "sigma_sq"  # "sigma_sq" | "two_sigma_sq"
    weight_by_fidelity: bool = False
    peer_normalization: str = "source_average"  # "source_average" | "mixing_weight"

    @classmethod
    def many_task(cls, dim: int = 10, target_similarity: float = 0.2) -> "ReasoningWeights":
        """The v2 recommended defaults, with bandwidth derived for ``dim``."""
        return cls(
            beta=2.0,
            lam=2.0,
            gamma=2.0,
            sigma_s=bandwidth_for_dimension(dim, target_similarity),
            metric="euclidean",
            kernel="sigma_sq",
            weight_by_fidelity=False,
            peer_normalization="source_average",
        )

    @classmethod
    def suzuki(cls) -> "ReasoningWeights":
        """The tuned Suzuki configuration: asymmetric weights, fidelity weighting, Hamming."""
        return cls(
            beta=2.0,
            lam=4.0,
            gamma=32.0,
            sigma_s=0.5,
            metric="hamming",
            kernel="two_sigma_sq",
            weight_by_fidelity=True,
            peer_normalization="mixing_weight",
        )


def peer_terms(
    candidate_embedding: Sequence[float],
    token_memory: Iterable[KnowledgeToken],
    mixing_weight: Callable[[str], float],
    weights: Optional[ReasoningWeights] = None,
) -> Tuple[float, float]:
    """Compute ``G_i(theta)`` (success attraction) and ``Lambda_i(theta)`` (failure avoidance).

        per source j:  [ sum_{k in K_j} w_k S(theta, theta_k) 1[s_k] ] / [ sum_{k in K_j} w_k ]
        w_k          =  c_k * eta_k   or   c_k        (weights.weight_by_fidelity)
        outer factor =  pi_ij         or   1 / |sources|   (weights.peer_normalization)

    ``mixing_weight`` maps a token's originating agent id to ``pi_ij``; normally
    ``partial(topology.uniform_weight, self.agent_id)``. A source whose weight is zero is
    dropped under **both** normalizations -- an agent should not be swayed by evidence
    arriving from outside its neighborhood, whichever way the remaining sources are scaled.

    The agent's own tokens participate here on equal footing with peers; the reference
    appends the agent's own token to its memory in the broadcast step for exactly that reason.
    """
    weights = weights or ReasoningWeights()

    by_source: Dict[str, List[KnowledgeToken]] = {}
    for token in token_memory:
        by_source.setdefault(token.provenance.agent_id, []).append(token)

    attraction = 0.0
    avoidance = 0.0
    n_sources = 0
    for source_id, tokens in by_source.items():
        if mixing_weight(source_id) <= 0.0:
            continue
        n_sources += 1
        outer = (
            mixing_weight(source_id)
            if weights.peer_normalization == "mixing_weight"
            else 1.0
        )
        token_weights = [
            t.advantage * (t.fidelity() if weights.weight_by_fidelity else 1.0)
            for t in tokens
        ]
        denom = sum(token_weights) + 1e-8
        source_attraction = 0.0
        source_avoidance = 0.0
        for token, weight in zip(tokens, token_weights):
            contribution = weight * similarity(
                candidate_embedding,
                token.embedding,
                weights.sigma_s,
                weights.metric,
                weights.kernel,
            )
            if token.signal is Signal.SUCCESS:
                source_attraction += contribution
            else:
                source_avoidance += contribution
        attraction += outer * source_attraction / denom
        avoidance += outer * source_avoidance / denom

    if weights.peer_normalization == "source_average" and n_sources:
        attraction /= n_sources
        avoidance /= n_sources
    return attraction, avoidance


def reasoning_score(
    posterior_mean: float,
    posterior_std: float,
    attraction: float,
    avoidance: float,
    weights: ReasoningWeights,
) -> float:
    """Eq. (1). Kept as a free function so it can be unit-tested against the paper directly."""
    return (
        posterior_mean
        + weights.beta * posterior_std
        + weights.lam * attraction
        - weights.gamma * avoidance
    )


def score_candidates(
    candidates: Sequence[Sequence[float]],
    posteriors: Sequence[Tuple[float, float]],
    token_memory: Sequence[KnowledgeToken],
    mixing_weight: Callable[[str], float],
    weights: Optional[ReasoningWeights] = None,
) -> Dict[int, float]:
    """Score a candidate batch, returning ``{candidate_index: R_i(theta)}``.

    ``candidates`` are embeddings; ``posteriors`` are the matching ``(mu, sigma)`` pairs from
    the agent's private surrogate.

    Note the reference standardizes ``mu`` and ``sigma`` before combining them with the peer
    terms (``mu_std``, ``sigma_std`` in ``run_suzuki.py``). That matters: ``G`` and ``Lambda``
    are normalized into [0, 1] by construction, so an unstandardized posterior on a 0-100
    yield scale would swamp them regardless of ``lam`` and ``gamma``. Standardization is the
    :class:`Surrogate` implementation's responsibility here -- it is the only component that
    knows the objective's scale.
    """
    weights = weights or ReasoningWeights()
    scores: Dict[int, float] = {}
    for idx, (embedding, (mu, sigma)) in enumerate(zip(candidates, posteriors)):
        attraction, avoidance = peer_terms(
            embedding, token_memory, mixing_weight, weights
        )
        scores[idx] = reasoning_score(mu, sigma, attraction, avoidance, weights)
    return scores
