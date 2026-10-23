"""ADKO v2 knowledge tokens -- the only object permitted to cross an agent boundary.

Follows the settled v2 method (arm ``ADKO-v2f-odds-bit``) and its reference implementation.
A v2 token is::

    k_i^t = { x , q , z , flag }

    x     the input as sent: phi(theta) after the problem's privacy channel. Suzuki flips each
          category to another option with probability 0.04365; the agent's own GP keeps the
          true point.
    q     my result's robust score against my own running median, clipped to [-1, 1] and
          quantized to a codebook every agent shares::

              a = clip((y - median(Y_i)) / (1.4826 * MAD(Y_i)), -1, 1)
              q = round(L * a) / L                 L = 127, i.e. 8 bits

          ``Y_i`` includes ``y`` itself. q = +1 exactly when a result is at least one robust SD
          above the lab's own median.
    z     optional text: the one-line hypothesis an L3 planner attached to the experiment it
          chose, with any percentage masked. The score never reads it.
    flag  the best-so-far bit: set when this result beat everything the sender ran before.
          Receivers keep each peer's newest flagged token past FIFO.

What changed from v1 (``../adko``): the binary signal ``s`` and the advantage ``c`` are folded
into one signed, multi-bit ``q`` against a *running* baseline, so there is no fixed threshold to
pick and no fidelity ``eta`` to carry; and the token goes out once a round, flagged rather than
re-sent when it is the sender's new best.

As in v1, what is *absent* is the point: no raw ``y``, no raw ``theta``, no GP parameter. The
baseline that turns ``y`` into ``q`` is the sender's own and never leaves it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: This round's result.
TAG_RESULT = 0
#: This round's result, and the sender's new best. The reference reserves ``1`` for the
#: re-sent incumbent token the settled method dropped, so the bit is ``2`` there too and
#: run records stay comparable.
TAG_BEST = 2

#: Consistency constant: 1.4826 * MAD estimates the SD of a normal sample.
MAD_TO_SIGMA = 1.4826

#: ``q_bits -> (kind, levels)``. The reference's ``--q-bits`` table; 8 bits is settled.
Q_BITS: Dict[int, Tuple[str, int]] = {
    1: ("sign", 7),
    2: ("uniform", 1),
    3: ("uniform", 3),
    4: ("uniform", 7),
    8: ("uniform", 127),
    32: ("float", 7),
}


@dataclass
class Provenance:
    """Who produced this token and when.

    ``agent_id`` is the APPFL node id, which the transport routes and meters by. ``source`` is
    the sender's index in the federation -- what receivers key their memory and trust on, and
    what the token's public id ``a{source}r{round}`` is built from.
    """

    agent_id: str = ""
    source: int = 0
    round: int = 0


@dataclass
class KnowledgeToken:
    """One v2 token ``k = {x, q, z, flag}``.

    Args:
        x: the shared input, ``phi(theta)`` after the privacy channel.
        q: the quantized robust score in ``[-1, 1]``.
        insight: optional text ``z``.
        tag: :data:`TAG_RESULT`, or :data:`TAG_BEST` when the best-so-far bit is set.
        provenance: node id, federation index and round.
    """

    x: List[float] = field(default_factory=list)
    q: float = 0.0
    insight: Optional[str] = None
    tag: int = TAG_RESULT
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        if not -1.0 <= self.q <= 1.0:
            raise ValueError(f"q must be in [-1, 1], got {self.q}")

    @property
    def source(self) -> int:
        return self.provenance.source

    @property
    def round(self) -> int:
        return self.provenance.round

    @property
    def is_best(self) -> bool:
        """The best-so-far bit."""
        return self.tag == TAG_BEST

    @property
    def id(self) -> str:
        """Public id, ``a{source}r{round}``; what an L3 planner cites as evidence."""
        return f"a{self.source}r{self.round}" + ("b" if self.tag == 1 else "")

    @property
    def key(self) -> Tuple[int, int, int]:
        """Identity for deduplication: a token arriving twice is stored once."""
        return (self.source, self.round, self.tag)

    # -- wire format ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def serialize(self) -> bytes:
        """Compact JSON, for logs and for metering, as in v1.

        The transports themselves send the token pickled by ``torch.save`` (hence the safe
        globals the package registers); this form is what the budget and the meter count.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str).encode()

    @classmethod
    def deserialize(cls, raw: bytes) -> "KnowledgeToken":
        d = json.loads(raw.decode())
        return cls(
            x=list(d.get("x", [])),
            q=float(d["q"]),
            insight=d.get("insight"),
            tag=int(d.get("tag", TAG_RESULT)),
            provenance=Provenance(**d["provenance"]),
        )

    def size_bits(self) -> int:
        """Size against the communication budget: the JSON form's, as v1 measures its tokens.

        Neither the method's own payload (the input's bits, 8 for ``q``, any text) nor the
        pickled bytes a transport actually sends, but one measure every transport shares, so
        their totals stay comparable.
        """
        return len(self.serialize()) * 8


class Quantizer:
    """The shared codebook for ``q``: levels ``k / L`` on ``[-1, 1]``, a sign, or float32.

    Every agent must use the same one, because a receiver reads ``q`` against the codebook it
    assumes the sender used.
    """

    def __init__(self, bits: int = 8):
        if int(bits) not in Q_BITS:
            raise ValueError(f"q_bits must be one of {sorted(Q_BITS)}, got {bits}")
        self.kind, self.levels = Q_BITS[int(bits)]

    @property
    def bits(self) -> int:
        if self.kind == "float":
            return 32
        if self.kind == "sign":
            return 1
        return int(math.ceil(math.log2(2 * self.levels + 1)))

    def __call__(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        if self.kind == "float":
            out = a.astype(np.float32).astype(float)
        elif self.kind == "sign":
            out = np.where(a >= 0.0, 1.0, -1.0)
        else:
            out = np.rint(a * self.levels) / self.levels
        out = out + 0.0  # drop negative zero
        return float(out) if out.ndim == 0 else out


def running_reference(history: Sequence[float]) -> Tuple[float, float]:
    """``(b, s)``: my running median and 1.4826 * MAD, or the SD (then 1) if the MAD vanishes.

    Computed with numpy exactly as the reference does: the fallback triggers whenever more
    than half of the history is one value -- common in Suzuki, where many conditions yield
    nothing -- and a differently summed SD would move ``q`` in its last bit.
    """
    y = np.asarray(history, dtype=float)
    b = float(np.median(y))
    s = float(np.median(np.abs(y - b))) * MAD_TO_SIGMA
    if s < 1e-8:
        s = float(np.std(y)) or 1.0
    return b, s


def relative_output(y: float, b: float, s: float, maximize: bool = True) -> float:
    """Signed advantage over the baseline, ``clip((y - b) / s, -1, 1)``."""
    a = (float(y) - float(b)) / max(float(s), 1e-8)
    a = a if maximize else -a
    return float(np.clip(a, -1.0, 1.0))


def score_against(
    value: float,
    history: Sequence[float],
    quantizer: Quantizer,
    maximize: bool = True,
) -> float:
    """``q`` of ``value`` against my history -- for a result, or for a prediction.

    The same function encodes my own results and the predictions trust compares a peer's
    tokens with, so the two are on one scale by construction.
    """
    b, s = running_reference(history)
    return float(quantizer(relative_output(value, b, s, maximize)))


def encode_token(
    *,
    agent_id: str,
    source: int,
    round: int,
    x_shared: Sequence[float],
    y: float,
    history: Sequence[float],
    quantizer: Quantizer,
    maximize: bool = True,
    insight: Optional[str] = None,
    new_best: bool = False,
) -> KnowledgeToken:
    """Algorithm 1 step 10 in v2: one result in, one token out.

    ``y`` is consumed here and never stored on the token; ``history`` must already include it,
    so a result takes part in its own median, as in the reference.
    """
    x = np.asarray(x_shared).tolist()
    return KnowledgeToken(
        x=list(x),
        q=score_against(y, history, quantizer, maximize),
        insight=insight,
        tag=TAG_BEST if new_best else TAG_RESULT,
        provenance=Provenance(
            agent_id=str(agent_id), source=int(source), round=int(round)
        ),
    )
