"""ADKO knowledge tokens -- the only object permitted to cross an agent boundary.

Follows Rillo et al., *ADKO: Agentic Decentralized Knowledge Optimization* (arXiv:2605.07863)
Section 4, and both reference implementations at ``github.com/lucasrillo/adko``
(``scientific_discovery/`` and the newer ``many_task/``). A token is::

    k_i^t = { s_i^t , c_i^t , z_i^t , phi(theta_i^t) }

    s   directional signal, SUCCESS iff y >= b, for a contextual baseline b that may be
        fixed or running (see the ``baseline`` package). Binary -- this quantization is the
        *primary source of compression loss*.
    c   advantage score in [0, 1], c = clip(|y - b| / scale, 0, 1). How decisively the
        outcome supports its signal. Near-baseline outcomes carry almost no information;
        extreme outcomes are nearly lossless.
    z   optional LM-generated natural-language insight (e.g., "high temperature destabilizes this
        material class"). Semantic carrier, not a number.
    phi embedding of the design point -- noised and/or quantized, non-invertible.

Note the ``many_task`` token carries no fidelity value at all and no LM insight; ``eta`` is
recomputed from ``c`` where needed, and ``z`` stays optional. Nothing here is transmitted that
either implementation would not transmit.

Note what is *absent*: the raw observation ``y``, the raw design point ``theta``, and any GP
parameter or sufficient statistic. ADKO Constraint 3.1 (Hard Privacy) forbids transmitting
any of them, so they have no field here. The reference keeps ``y_raw`` and ``theta_true`` as
local-only fields stripped by a ``redact()`` call; leaving them out entirely means a missed
call cannot leak.

Module named ``knowledge_token`` rather than ``token`` on purpose: a top-level ``token.py``
shadows the standard library's ``token`` module for any interpreter started with this
directory on ``sys.path``, which breaks ``traceback`` and therefore everything.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Signal(str, Enum):
    """ADKO's binary directional signal ``s``."""

    SUCCESS = "success"
    FAIL = "fail"


@dataclass
class Provenance:
    """Who produced this token and when. ``round`` is ADKO's ``t``, read by the pruner."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    round: int = 0


def binary_entropy(p: float) -> float:
    """H_b(p) in bits, with the usual 0 log 0 = 0 convention.

    Returns the entropy of a Bernoulli r.v. with success of probability p.

    This is used in the KnowledgeToken.fidelity() method.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class KnowledgeToken:
    """One ADKO token. Constructed by an agent, routed by the transport layer.

    The transport treats this as opaque except for :meth:`size_bits` (Constraint 3.2 budget
    accounting) and ``provenance.agent_id`` (routing). Interpreting ``insight`` or using
    ``embedding`` in a similarity kernel is the agent layer's business.
    """

    signal: Signal
    advantage: float  # c in [0, 1]
    embedding: List[float] = field(default_factory=list)  # phi(theta)
    insight: Optional[str] = None  # z, optional LM text
    insight_model: Optional[str] = None  # which LM wrote z, for the bias/noise study
    space_id: str = ""
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        if not 0.0 <= self.advantage <= 1.0:
            raise ValueError(f"advantage score must be in [0, 1], got {self.advantage}")

    # -- ADKO Definition 3 / Algorithm 2 ----------------------------------------------

    # Definition: Token Fidelity. The fidelity of the token k is the fraction of mutual information about the true outcome that survives binary quantization: n_k = I(f_j(\theta_k); k) / H(f_j(\theta_k)) \in [0,1], where I(\cdot;\cdot) denotes mutual information and H(\cdot) denotes differential entropy under the Gaussian Process posterior.
    def fidelity(self) -> float:
        """Estimated token fidelity: ``eta = c * (1 - H_b((1 - c) / 2))``.

        The fraction of mutual information about the true outcome that survives binary
        quantization. ``c = 1`` (outcome far from the threshold) gives ``eta = 1``, nearly
        lossless; ``c = 0`` (outcome sitting on the threshold) gives ``eta = 0``, the signal
        is a coin flip and the token says nothing (maximum entropy).
        """
        # \eta(c) = c \times (1 - H_b((1 - c) / 2)), where \eta is the fidelity and H_b is the binary entropy
        return self.advantage * (1.0 - binary_entropy((1.0 - self.advantage) / 2.0))

    # Algorithm 2: Fidelity-Aware Token Pruning --- Agent i, Budget B
    # Input: Token memory K_i^t with |K_i^t| > B, recency weight \alpha_\tau > 0
    # Output: Pruned K_i^t with |K_i^t| = B.
    # 1. while |K_i^t| > B do
    # 2.       for each token k \in K_i^t do
    # 3.            \hat{\eta}_k \leftarrow c_k \cdot (1 - H_b((1-c_k)/2)) --- estimate token fidelity
    # 4.            score(k) \leftarrow \hat{\eta}_k \cdot c_k \cdot exp(-\alpha_tau(t - k.round))
    # 5.       end for
    # 6.       drop \leftarrow \arg\min_{k \in K_i^t}score(k)
    # 7.       K_i^t \leftarrow K_i^t \ {dropped}
    # 8. end while
    def pruning_score(self, current_round: int, alpha_tau: float = 0.01) -> float:
        """Algorithm 2 line 4: ``score = eta * c * exp(-alpha_tau * (t - k.round))``."""
        age = max(0, current_round - self.provenance.round)
        return self.fidelity() * self.advantage * math.exp(-alpha_tau * age)

    # -- wire format ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def serialize(self) -> bytes:
        """Wire format, shared by every transport backend.

        JSON keeps tokens inspectable in logs and portable across the MPI and gRPC paths.
        Tokens are hundreds of bytes, so the encoding overhead is irrelevant next to the
        six orders of magnitude this whole approach saves over model sharing.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str).encode()

    @classmethod
    def deserialize(cls, raw: bytes) -> "KnowledgeToken":
        d = json.loads(raw.decode())
        return cls(
            signal=Signal(d["signal"]),
            advantage=d["advantage"],
            embedding=d.get("embedding", []),
            insight=d.get("insight"),
            insight_model=d.get("insight_model"),
            space_id=d.get("space_id", ""),
            provenance=Provenance(**d["provenance"]),
        )

    def size_bits(self) -> int:
        """Token size against Constraint 3.2 (at most B bits per neighbor per round).

        Measured on the serialized form, so it includes the natural-language insight --
        which dominates. A numeric-only token is a few hundred bits; one carrying a sentence
        of LM insight is a few thousand. That asymmetry is why ``insight`` is optional, and
        why the paper reports 333 B/round for ADKO against 232.7 MB for FedAvg BO.
        """
        return len(self.serialize()) * 8


# Definition: Knowledge Token. k_i^t = \{s_i^t, c_i^t, z_i^t, \varphi(\theta_i^t) \} where s_i^t \in \{success, fail\}. In particular s_i^t = success iff y_i^t \ge b_t^i, where b_t^i > 0is the contextual baseline
def encode_token(
    *,
    agent_id: str,
    round: int,
    observation: float,
    threshold: float,
    scale: float,
    embedding: Sequence[float],
    space_id: str = "",
    insight: Optional[str] = None,
    insight_model: Optional[str] = None,
    objective: str = "maximize",
) -> KnowledgeToken:
    """ADKO Algorithm 1 step 10, ``M_i.ENCODE(theta, y, tau)``, without the LM part.

    ``observation`` is consumed here and never stored -- this function is the privacy
    boundary. Everything downstream sees only the sign and the normalized magnitude::

        s = SUCCESS  iff  y >= threshold      (maximize; reversed when minimizing)
        c = clip(|y - threshold| / scale, 0, 1)

    ``threshold`` is ADKO's ``b`` and ``scale`` normalizes the deviation. Both come from a
    :class:`~appfl.decentralized.algorithm.adko.baseline.Baseline`, which may be fixed or updated from local history. The paper (v2) is explicit that either is legitimate: a fixed
    threshold when "success" has a natural domain meaning, a running median otherwise. See
    ``baseline/`` for why the choice matters rather than being a detail.
    """
    deviation = (
        observation - threshold if objective == "maximize" else threshold - observation
    )  # computes deviation d = y - b (max) or d = b - y (min) for calculating the advantage score
    advantage = (
        0.0 if scale <= 0 else min(1.0, abs(deviation) / scale)
    )  # c_i^t = |y_i^t - b_t^i|/(scale), the scale is either ||y - b_t^i||_{\max} which is described in the paper or another "bounded, monotone confidence formula based on b_t^i". The paper used for their FMTBO using the agent's running median and median absolute deviation.
    return KnowledgeToken(
        signal=Signal.SUCCESS if deviation >= 0 else Signal.FAIL,
        advantage=advantage,
        embedding=list(embedding),
        insight=insight,
        insight_model=insight_model,
        space_id=space_id,
        provenance=Provenance(agent_id=agent_id, round=round),
    )
