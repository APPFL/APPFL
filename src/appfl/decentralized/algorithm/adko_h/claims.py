"""Claims and verdicts: how campaigns coordinate without a coordinator.

A centralized method can divide the space with an allocator that sees everything. ADKO-H has
to do it with messages, because the point of the method is to still work at hundreds of agents
on a sparse graph, where no node can hold the global picture and the diameter is several hops.

Two messages do it, and both ride along with the ordinary knowledge token:

* a **claim**, sent when a campaign starts: *I am working this region with this rule until this
  round, on this hypothesis*. A neighbour does not obey it -- it adds a soft penalty to
  candidates inside an active foreign claim, so two agents converge on the same spot only when
  both scores insist. Claims expire by round, so a crashed or slow agent cannot fence off a
  region forever.
* a **verdict**, sent when the campaign closes: *that claim passed or failed, on this metric at
  this value*. A verdict is the only knowledge in ADKO-H that is worth more than one hop: it
  is a tested statement about a region, so it travels ``ttl`` hops with the origin preserved,
  deduplicated by ``(origin, claim_id)``.

Both are small, structured and already quantized to a region plus a number, which is also what
a randomized-response channel would need if differential privacy is added later; nothing here
carries a raw design point or a raw outcome.

One arm (``AgentConfig.share_detail``) widens the channel on purpose, to find out whether prose
is worth its bits: the full hypothesis rather than a truncation, the planner's own sentence
about what its last campaign taught it, and the box this campaign's hits fell in. The last of
those is still region language, so the invariant above holds -- a box over the planner's own
coordinates, not the molecule that was run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Default hops a claim and a verdict travel. One hop is ADKO v2's behaviour.
CLAIM_TTL = 1
VERDICT_TTL = 3
#: Rounds a claim outlives its budget by, covering a straggling agent.
CLAIM_SLACK = 2
#: Verdicts kept for the prompts, newest first.
VERDICT_KEEP = 10


#: Characters of text a claim and a verdict carry. The rest stays local: the channel is
#: measured against the token it rides on, and prose is what makes it heavy.
CLAIM_CHARS = 100
VERDICT_CHARS = 140
#: Characters allowed on the detailed channel (``AgentConfig.share_detail``), which trades
#: token size for prose: the full hypothesis, a sentence of the planner's own account of its
#: last campaign, and the box its hits fell in. Whether that prose is worth the bits is the
#: question the ``adkoh3-rich`` arm asks.
DETAIL_CHARS = 400
NOTE_CHARS = 240


@dataclass
class Claim:
    """An announced campaign. ``expires`` is an absolute round, so relaying cannot extend it."""

    origin: int = 0
    claim_id: str = ""
    region: Any = None
    rule: str = "adko"
    hypothesis: str = ""
    round: int = 0
    expires: int = 0
    ttl: int = CLAIM_TTL
    #: The planner's sentence to its neighbours, sent only on the detailed channel.
    note: str = ""

    @property
    def key(self) -> Tuple[int, str]:
        return (int(self.origin), str(self.claim_id))

    def to_dict(self) -> Dict[str, Any]:
        """The wire form. An empty note is left out, so a non-detailed claim weighs what it did."""
        d = asdict(self)
        if not d["note"]:
            d.pop("note")
        return d


@dataclass
class Verdict:
    """A closed campaign's outcome, as it travels. ``value`` may be ``None`` (an infinite metric)."""

    origin: int = 0
    claim_id: str = ""
    region: Any = None
    metric: str = ""
    value: Optional[float] = None
    op: str = ">="
    threshold: float = 0.0
    success: bool = False
    hypothesis: str = ""
    round: int = 0
    ttl: int = VERDICT_TTL
    #: What the campaign found, for arms that share it (``AgentConfig.share_hits``): its
    #: experiments, hits, hit families nobody had reported, and best score. A test outcome
    #: alone says whether a planner was right about its own claim, not where the hits are.
    spent: Optional[int] = None
    hits: Optional[int] = None
    hit_families: Optional[int] = None
    best: Optional[float] = None
    #: The detailed channel (``AgentConfig.share_detail``): the planner's own sentence about
    #: what it had just learned, and the box this campaign's hits actually fell in -- which is
    #: region language, the same vocabulary a claim already travels in, not a design point.
    note: str = ""
    hit_box: Any = None

    @property
    def key(self) -> Tuple[int, str]:
        return (int(self.origin), str(self.claim_id))

    def line(self, space=None) -> str:
        """One line for a prompt: what was tried, and whether it worked.

        With the campaign's findings on board, the line also says where it looked (when
        ``space`` is given) and what it found there.
        """
        v = "n/a" if self.value is None else f"{self.value:g}"
        test = (
            f"[a{self.origin}] {'HELD' if self.success else 'FAILED'}: {self.hypothesis or '(none)'} "
            f"({self.metric} {v} vs {self.op} {self.threshold:g})"
        )
        if self.hits is None:
            return test + self._detail(space)
        where = f" in {region_of(space, self.region).label(space)}" if space is not None else ""
        best = "n/a" if self.best is None else f"{self.best:.3g}"
        return (
            f"{test}; found{where}: {self.hits} hits and {self.hit_families} new hit families "
            f"in {self.spent} experiments, best score {best}" + self._detail(space)
        )

    def _detail(self, space=None) -> str:
        """The detailed channel's additions, when it is on: where the hits sat, and the note."""
        out = ""
        if self.hit_box is not None and space is not None:
            out += f"; its hits sat in {region_of(space, self.hit_box).label(space)}"
        if self.note:
            out += f'; a{self.origin} notes: "{self.note}"'
        return out

    def to_dict(self) -> Dict[str, Any]:
        """The wire form. Findings an arm does not share are left out, not sent as nulls."""
        d = asdict(self)
        for k in _FINDINGS:
            if d[k] is None:
                d.pop(k)
        if not d["note"]:
            d.pop("note")
        if d["hit_box"] is None:
            d.pop("hit_box")
        return d


_FINDINGS = ("spent", "hits", "hit_families", "best")


def region_of(space, data: Any):
    """Rebuild a region received over the wire, in either form, validated locally."""
    if isinstance(data, dict):
        return space.region(bounds=data.get("bounds"), picks=data.get("picks"))
    return space.unpack(data)


class ClaimStore:
    """One agent's view of who is working where, and what has been settled.

    Args:
        space: the region space, for masking and for rebuilding received regions.
        source: this agent's federation index, so its own claims are not penalized.
        keep: how many verdicts to keep for the prompts.
    """

    def __init__(self, space, source: int, keep: int = VERDICT_KEEP):
        self.space = space
        self.source = int(source)
        self.keep = int(keep)
        self.claims: Dict[Tuple[int, str], Claim] = {}
        self.verdicts: Dict[Tuple[int, str], Verdict] = {}
        self._order: List[Tuple[int, str]] = []
        self._out_claims: List[Claim] = []
        self._out_verdicts: List[Verdict] = []
        #: Counters for the run record: how much of the channel was used and reused.
        self.totals: Dict[str, int] = {
            "claims_in": 0,
            "claims_new": 0,
            "verdicts_in": 0,
            "verdicts_new": 0,
            "relayed": 0,
            "steered": 0,
        }

    # -- own messages ---------------------------------------------------------------------

    def announce(self, claim: Claim) -> None:
        """Queue this agent's own claim for the next token, and remember it locally."""
        self.claims[claim.key] = claim
        self._out_claims.append(claim)

    def settle(self, verdict: Verdict) -> None:
        self._remember_verdict(verdict)
        self._out_verdicts.append(verdict)

    def outbox(self) -> Tuple[List[Claim], List[Verdict]]:
        """Messages to attach to this round's token; clears the queue."""
        c, v = self._out_claims, self._out_verdicts
        self._out_claims, self._out_verdicts = [], []
        return c, v

    # -- received messages ----------------------------------------------------------------

    def ingest(self, claims: Iterable[Claim], verdicts: Iterable[Verdict]) -> None:
        """Store what is new and queue it for relay with one hop spent.

        A duplicate is dropped rather than forwarded, which is what keeps a dense
        neighbourhood from amplifying one claim into a flood.
        """
        for c in claims:
            self.totals["claims_in"] += 1
            if c.origin == self.source or c.key in self.claims:
                continue
            self.totals["claims_new"] += 1
            self.claims[c.key] = c
            if c.ttl > 1:
                self._out_claims.append(Claim(**{**c.to_dict(), "ttl": c.ttl - 1}))
                self.totals["relayed"] += 1
        for v in verdicts:
            self.totals["verdicts_in"] += 1
            if v.origin == self.source or v.key in self.verdicts:
                continue
            self.totals["verdicts_new"] += 1
            self._remember_verdict(v)
            if v.ttl > 1:
                self._out_verdicts.append(Verdict(**{**v.to_dict(), "ttl": v.ttl - 1}))
                self.totals["relayed"] += 1

    def _remember_verdict(self, v: Verdict) -> None:
        if v.key not in self.verdicts:
            self._order.append(v.key)
        self.verdicts[v.key] = v
        while len(self._order) > self.keep:
            self.verdicts.pop(self._order.pop(0), None)

    # -- what the agent reads -------------------------------------------------------------

    def active(self, now: int) -> List[Claim]:
        """Foreign claims still in force at round ``now``."""
        return [c for c in self.claims.values() if c.origin != self.source and now <= c.expires]

    def penalty(self, F: np.ndarray, now: int) -> np.ndarray:
        """1.0 for a candidate inside any active foreign claim, else 0.0.

        An indicator rather than a count: being in two neighbours' regions is not twice as bad,
        and a sum would make the penalty depend on how many neighbours a node happens to have,
        which is exactly the dependence a sparse-graph method must not have.
        """
        F = np.atleast_2d(np.asarray(F, dtype=float))
        out = np.zeros(len(F))
        for c in self.active(now):
            out = np.maximum(out, region_of(self.space, c.region).mask(self.space, F).astype(float))
        hit = int(np.count_nonzero(out))
        self.totals["steered"] += hit
        return out

    def recent_verdicts(self, n: Optional[int] = None) -> List[Verdict]:
        keys = self._order[-(n or self.keep) :]
        return [self.verdicts[k] for k in keys if k in self.verdicts][::-1]

    def digest(self, n: Optional[int] = None) -> str:
        """The verdict block for a prompt: what peers have tested, newest first."""
        lines = [v.line(self.space) for v in self.recent_verdicts(n)]
        return "\n".join(lines) if lines else "(nothing yet)"

    def claim_digest(self, now: int) -> str:
        """The active-claim block: where peers are working right now."""
        rows = [
            f"[a{c.origin}] {region_of(self.space, c.region).label(self.space)} "
            f"rule={c.rule} until round {c.expires}: {c.hypothesis or '(none)'}"
            + (f'; a{c.origin} notes: "{c.note}"' if c.note else "")
            for c in self.active(now)
        ]
        return "\n".join(rows) if rows else "(none)"
