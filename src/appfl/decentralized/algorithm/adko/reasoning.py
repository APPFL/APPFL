"""The ADKO reasoning score -- Eq. (1), the thing each agent argmaxes to pick its next probe.

    R_i(theta) = U_i(theta)  +  beta * sigma_i(theta)  +  lambda * G_i(theta)  -  gamma * Lambda_i(theta)
                 \\_________/     \\________________/       \\______________/         \\__________________/
                  what I expect   how unsure I am          do my neighbours         do my neighbours
                  from my data    personally               succeed here?            fail here?

The first two terms, U_i(theta) and sigma_i(theta), are exactly GP-UCB over the agent's *private* posterior. The last two are
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

# Similarity Kernel: S(\theta, \theta_k) = \exp(- d(\varphi(\theta), \varphi(\theta_k))^2 / denom), where denom is the bandwidth-dependent normalizer.
# similarity() computes the above.


def distance(
    a: Sequence[float], b: Sequence[float], metric: str = "euclidean"
) -> float:
    """Return distance between embeddings.

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
    """Return ``sigma_s = sqrt((dim / 6) / -log(target_similarity))``."""
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
    """Return ``S(a,b) = exp(-d(a,b)^2 / denom)``.

    Args:
        sigma_s: Controls how quickly similarity decays.
        metric: Distance type.
        kernel: Denominator choice, ``sigma_s^2`` or ``2 * sigma_s^2``.
    """
    if not embedding_a or not embedding_b:
        return 0.0
    d = distance(embedding_a, embedding_b, metric)
    bw = max(sigma_s, 1e-12)
    denom = bw**2 if kernel == "sigma_sq" else 2 * bw**2
    return math.exp(-(d**2) / denom)


@dataclass
class ReasoningWeights:
    """Hyperparameters for ``mu + beta*sigma + lam*G - gamma*Lambda``.

    Args:
        beta: Weight on uncertainty ``sigma``.
        lam: Weight on success term ``G``.
        gamma: Weight on failure term ``Lambda``.
        sigma_s: Similarity bandwidth.
        metric: Distance type.
        kernel: Similarity denominator choice.
        weight_by_fidelity: Use ``c * eta`` instead of ``c``.
        peer_normalization: Combine peers by average or graph weight.
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
    def many_task(
        cls, dim: int = 10, target_similarity: float = 0.2
    ) -> "ReasoningWeights":
        """Return the Rillo et al. (ADKO) v2 defaults."""
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
        """Return the Suzuki benchmark defaults."""
        # intended to replicate the suzuki benchmark configuration.
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
    """Return peer attraction ``G`` and avoidance ``Lambda``.

    For each source, tokens contribute ``w_k S(theta, theta_k)`` where
    ``w_k = c_k * eta_k`` when fidelity weighting is enabled, otherwise ``c_k``.
    Success tokens add to ``G``; failure tokens add to ``Lambda``.

    Args:
        candidate_embedding: Candidate as ``phi(theta)``.
        token_memory: Tokens this agent has.
        mixing_weight: Returns neighbor weight ``pi_ij``.
        weights: Scoring settings.
    """
    weights = weights or ReasoningWeights()

    by_source: Dict[str, List[KnowledgeToken]] = (
        {}
    )  # group tokens by their respective agent --- matches the per-neighbor source aggretation idea where success and failure terms are computed per neighbor then aggregated over neighbors.
    for token in token_memory:
        by_source.setdefault(token.provenance.agent_id, []).append(token)

    attraction = 0.0  # G_i(\theta)
    avoidance = 0.0  # Lambda_i(\theta)
    n_sources = 0  # used only for "source_average" normalization

    # for each neighbor/source j compute its contribution to both successes and failures
    for source_id, tokens in by_source.items():

        # drop the sources that are not neighbors in the graph, satisfying the paper's neighborhood (N_i) assumption
        if mixing_weight(source_id) <= 0.0:
            continue

        # count sources and set outer normalization factor
        n_sources += 1
        # for peer_normalization = "mixing_weight" we scale by pi_ij (graph-dependent mixing)
        # for peer_normalization = "source_average" you scale by 1.0 and later average across sources.
        outer = (
            mixing_weight(source_id)
            if weights.peer_normalization == "mixing_weight"
            else 1.0
        )
        # compute token weights inside the source
        token_weights = [
            t.advantage * (t.fidelity() if weights.weight_by_fidelity else 1.0)
            for t in tokens
        ]

        # compute the denom for within source normalization
        denom = sum(token_weights) + 1e-8

        # init per-source accumulators
        source_attraction = 0.0
        source_avoidance = 0.0
        for token, weight in zip(tokens, token_weights):
            # compute kernel similarity weighted contribution
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
        # normalize source and global aggregation, which is the per-source normalized sum then aggregate across sources
        attraction += outer * source_attraction / denom
        avoidance += outer * source_avoidance / denom

    # averages the per-source normalization contributions across all neighbor sources that contributed
    if weights.peer_normalization == "source_average" and n_sources:
        attraction /= n_sources
        avoidance /= n_sources
    return attraction, avoidance


def peer_terms_batch(
    candidate_embeddings: Sequence[Sequence[float]],
    token_memory: Sequence[KnowledgeToken],
    mixing_weight: Callable[[str], float],
    weights: Optional[ReasoningWeights] = None,
):
    """Vectorized :func:`peer_terms` for many candidates.

    Args:
        candidate_embeddings: Candidates as ``phi(theta)``.
        token_memory: Tokens this agent has.
        mixing_weight: Returns neighbor weight ``pi_ij``.
        weights: Scoring settings.
    """
    import numpy as np

    weights = weights or ReasoningWeights()
    n = len(candidate_embeddings)
    attraction = np.zeros(n)
    avoidance = np.zeros(n)
    if n == 0 or not token_memory:
        return attraction, avoidance

    usable = [t for t in token_memory if len(t.embedding) > 0]
    if not usable:
        return attraction, avoidance

    candidates = np.asarray(candidate_embeddings, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates.reshape(n, -1)

    by_source: Dict[str, List[KnowledgeToken]] = {}
    for token in usable:
        by_source.setdefault(token.provenance.agent_id, []).append(token)

    bw = max(weights.sigma_s, 1e-12)
    denom_kernel = bw**2 if weights.kernel == "sigma_sq" else 2 * bw**2
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

        embeddings = np.asarray([t.embedding for t in tokens], dtype=float)
        if embeddings.shape[1] != candidates.shape[1]:
            raise ValueError(
                f"token embedding dim {embeddings.shape[1]} != candidate dim "
                f"{candidates.shape[1]} for source {source_id!r}; the federation must share "
                f"one phi"
            )

        if weights.metric == "hamming":
            distances = (candidates[:, None, :] != embeddings[None, :, :]).mean(axis=2)
        else:
            diff = candidates[:, None, :] - embeddings[None, :, :]
            distances = np.sqrt((diff**2).sum(axis=2))

        kernel = np.exp(-(distances**2) / denom_kernel)  # (n_candidates, n_tokens)

        token_weights = np.asarray(
            [
                t.advantage * (t.fidelity() if weights.weight_by_fidelity else 1.0)
                for t in tokens
            ],
            dtype=float,
        )
        is_success = np.asarray(
            [1.0 if t.signal is Signal.SUCCESS else 0.0 for t in tokens], dtype=float
        )

        weighted = kernel * token_weights[None, :]
        denom = float(token_weights.sum()) + 1e-8
        attraction += outer * (weighted * is_success[None, :]).sum(axis=1) / denom
        avoidance += (
            outer * (weighted * (1.0 - is_success)[None, :]).sum(axis=1) / denom
        )

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
    """Return ``mu + beta*sigma + lam*G - gamma*Lambda``."""
    return (
        posterior_mean
        + weights.beta * posterior_std
        + weights.lam * attraction
        - weights.gamma * avoidance
    )


def score_candidates(
    candidates: Sequence[
        Sequence[float]
    ],  # list of candidate points represented as embeddings (\phi(\theta))
    posteriors: Sequence[Tuple[float, float]],  # list of (\mu, \sigma)
    token_memory: Sequence[KnowledgeToken],  # K_i^t
    mixing_weight: Callable[[str], float],  # \pi_{ij}
    weights: Optional[ReasoningWeights] = None,  # hyperparam presets
) -> Dict[int, float]:
    """Return ADKO scores keyed by candidate index.

    Args:
        candidates: Candidate embeddings.
        posteriors: Matching ``(mu, sigma)`` values.
        token_memory: Current tokens ``K_i^t``.
        mixing_weight: Returns neighbor weight ``pi_ij``.
        weights: Scoring settings.
    """
    weights = weights or ReasoningWeights()

    try:
        attractions, avoidances = peer_terms_batch(
            candidates, token_memory, mixing_weight, weights
        )
        return {
            idx: reasoning_score(mu, sigma, attractions[idx], avoidances[idx], weights)
            for idx, (mu, sigma) in enumerate(posteriors)
        }
    except ImportError:
        pass

    scores: Dict[int, float] = {}
    for idx, (embedding, (mu, sigma)) in enumerate(zip(candidates, posteriors)):
        attraction, avoidance = peer_terms(
            embedding, token_memory, mixing_weight, weights
        )
        scores[idx] = reasoning_score(mu, sigma, attraction, avoidance, weights)
    return scores
