"""The ADKO-H token: a v2 token plus the campaign channel.

A v2 token is ``{x, q, z, flag}`` and nothing else, which is the right payload when every agent
is chasing the same scalar. A diversity problem needs two more things, and both are small:

* **where the result landed**, as a quantized descriptor and an optional family key. Without
  it, novelty is measured against my own history only, and ten agents rediscover the same
  class ten times -- the exact failure the centralized allocator exists to prevent. With it,
  each agent's novelty rule sees the team's footprint at the cost of a few bytes a round.
* **claims and verdicts**, the campaign channel from ``claims.py``, which ride along rather
  than opening a second connection: one token a round remains the only thing crossing an agent
  boundary, so the communication constraint that defines ADKO still holds.

Both additions are structured and already quantized, so a randomized-response channel can be
put in front of them later without touching this file. Sharing either is optional, because
"does sharing the footprint explain the gain?" is one of the ablations the screen runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from appfl.decentralized.algorithm.adko_h.claims import Claim, Verdict
from appfl.decentralized.algorithm.adko_v2.knowledge_token import (
    TAG_BEST,
    TAG_RESULT,
    KnowledgeToken,
    Provenance,
    Quantizer,
)

#: Levels used for a shared descriptor component, matching the 8-bit codebook ``q`` uses.
DESCRIPTOR_LEVELS = 127


def quantize_descriptor(z: Sequence[float], scale: Sequence[float]) -> List[float]:
    """A descriptor on the shared 8-bit codebook, component by component.

    ``scale`` is the run's frozen descriptor scale (one positive number per component), so
    every agent quantizes onto the same lattice; a descriptor outside the scale is clipped,
    which costs resolution at the edges and keeps the codebook shared.
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(scale, dtype=float)
    if z.size == 0 or s.size != z.size:
        return []
    a = np.clip(z / np.where(s > 0, s, 1.0), -1.0, 1.0)
    return (np.rint(a * DESCRIPTOR_LEVELS) / DESCRIPTOR_LEVELS).astype(float).tolist()


@dataclass
class HToken(KnowledgeToken):
    """``{x, q, z, flag}`` plus ``descriptor``, ``family``, ``claims`` and ``verdicts``."""

    descriptor: List[float] = field(default_factory=list)
    family: str = ""
    claims: List[Claim] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c if isinstance(c, dict) else c.to_dict() for c in self.claims]
        d["verdicts"] = [v if isinstance(v, dict) else v.to_dict() for v in self.verdicts]
        return d

    @classmethod
    def deserialize(cls, raw: bytes) -> "HToken":
        d = json.loads(raw.decode())
        return cls(
            x=list(d.get("x", [])),
            q=float(d["q"]),
            insight=d.get("insight"),
            tag=int(d.get("tag", TAG_RESULT)),
            provenance=Provenance(**d["provenance"]),
            descriptor=list(d.get("descriptor", [])),
            family=str(d.get("family", "")),
            claims=[Claim(**c) for c in d.get("claims", [])],
            verdicts=[Verdict(**v) for v in d.get("verdicts", [])],
        )

    def payload_bits(self) -> int:
        """The method's own payload, as the paper counts it, not the JSON envelope.

        ``len(x)`` inputs and each descriptor component at 8 bits, 8 for ``q``, 1 for the flag,
        and the text and campaign messages at one byte per character. What ``size_bits``
        reports stays the JSON length, so budgets remain comparable with v1 and v2.
        """
        bits = 8 * (len(self.x) + len(self.descriptor)) + 8 + 1
        bits += 8 * len(self.insight or "") + 8 * len(self.family)
        for m in list(self.claims) + list(self.verdicts):
            bits += 8 * len(json.dumps(m if isinstance(m, dict) else m.to_dict(), separators=(",", ":")))
        return bits


def encode(
    *,
    agent_id: str,
    source: int,
    round: int,
    x_shared: Sequence[float],
    u: float,
    history: Sequence[float],
    quantizer: Quantizer,
    new_best: bool = False,
    insight: Optional[str] = None,
    descriptor: Optional[Sequence[float]] = None,
    family: str = "",
    claims: Sequence[Claim] = (),
    verdicts: Sequence[Verdict] = (),
) -> HToken:
    """One result and this round's campaign messages in, one token out.

    ``history`` must already include ``u``, so a result takes part in its own median exactly as
    in v2 -- the quantized ``q`` has to mean the same thing in both methods for the baselines
    to be comparable.
    """
    from appfl.decentralized.algorithm.adko_v2.knowledge_token import score_against

    return HToken(
        x=list(np.asarray(x_shared, dtype=float).tolist()),
        q=score_against(u, history, quantizer, maximize=True),
        insight=insight,
        tag=TAG_BEST if new_best else TAG_RESULT,
        provenance=Provenance(agent_id=str(agent_id), source=int(source), round=int(round)),
        descriptor=list(descriptor or []),
        family=str(family or ""),
        claims=list(claims),
        verdicts=list(verdicts),
    )
