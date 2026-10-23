"""The ADKO-H agent: campaigns outside the loop, a numeric rule inside it.

One round, for one agent:

1. **plan if needed.** With no campaign, or with the last one spent, the agent assembles its
   context -- its own history, the active claims, the verdicts that reached it, its territory --
   and the driver asks the planner for a design. Every agent needing a design in a round is
   asked in one batched call, which is what keeps a 27B model serving a hundred agents.
2. **pick.** The problem draws a candidate pool; the agent scores it with its GP, the peer
   field and the round's ``match_rms`` factor; the campaign's rule turns that into values; the
   claim penalty and any novelty bonus adjust them; the argmax *inside the campaign's region*
   is the experiment. The planner is not consulted here, ever.
3. **complete.** The result updates the GP and the history, and counts against the campaign's
   budget. On the campaign's last experiment the test is evaluated and the verdict written --
   the verdict is computed from data, not asserted by the model.
4. **emit and observe.** One token goes out, carrying ``q``, optionally where the result landed,
   and this round's claims and verdicts. Incoming tokens update trust, memory, and the claim
   store, and new claims and verdicts are queued for relay.

What this does *not* do is pick experiments with a language model. The L3 screen measured that
and it lost to the score it was replacing; the same calls spent one level up buy a region, a
rule, a budget and a falsifiable test instead, and every experiment stays with the calibrated
rule that won.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h import metrics as hmetrics
from appfl.decentralized.algorithm.adko_h import rules as hrules
from appfl.decentralized.algorithm.adko_h.campaign import (
    BUDGET_RANGE,
    Campaign,
    Evidence,
    Test,
    random_design,
)
from appfl.decentralized.algorithm.adko_h.claims import (
    CLAIM_CHARS,
    CLAIM_SLACK,
    CLAIM_TTL,
    VERDICT_CHARS,
    VERDICT_TTL,
    Claim,
    ClaimStore,
    Verdict,
)
from appfl.decentralized.algorithm.adko_h.problem import Candidates, Outcome, Pick, Problem
from appfl.decentralized.algorithm.adko_h.token import HToken, encode, quantize_descriptor
from appfl.decentralized.algorithm.adko_v2.components import Surrogate
from appfl.decentralized.algorithm.adko_v2.knowledge_token import Quantizer, score_against
from appfl.decentralized.algorithm.adko_v2.memory import TokenMemory
from appfl.decentralized.algorithm.adko_v2.reasoning import (
    Calibrator,
    PeerField,
    ReasoningWeights,
    score,
)
from appfl.decentralized.algorithm.adko_v2.trust import TrustState, agreement, relative_sigma


@dataclass
class AgentConfig:
    """Everything an arm varies. The defaults are ADKO v2's settled method plus campaigns.

    Args:
        weights: the v2 score's ``beta``, ``lam``, ``gamma``, kernel and bandwidth.
        q_bits: token codebook; 8 is settled.
        memory_rounds: FIFO depth, in rounds times degree.
        trust: whether source weights follow the odds of learned trust.
        warmup_rounds: random picks before the first campaign, so the GP has data.
        pool_size: candidates drawn per agent-round. The reference set for ``match_rms`` too.
        claim_weight: strength of the steer-away penalty, in units of ``RMS(sigma)``.
        novelty_weight: novelty bonus applied whatever the rule -- the diversity-aware
            baseline is this with no planner attached.
        share_descriptor: put the quantized descriptor on the token.
        share_family: put the family key on the token.
        claims_on: announce claims and relay verdicts at all (off is the coordination ablation).
        share_text: carry the hypothesis on the wire. Off by default: a relayed claim's prose
            costs several times the result it rides on (measured), and only the verdict's text
            earns its place in a prompt, so verdicts keep a short form either way.
        territory: confine the first campaign to the agent's own Voronoi cell.
        default_budget: campaign length when there is no planner (the plain-ADKO arm runs one
            open-ended campaign instead).
        force_rule: the rule the no-planner arm uses for its open-ended campaign, and, when set,
            the rule of every planned campaign too. How much the endpoint moves across the five
            rules is how much headroom a planner that chooses rules has at all; if they all
            score alike, ADKO-H cannot win by choosing between them and the honest thing is to
            know that before spending a GPU on it. Fixing it under a planner keeps that planner's
            regions and takes the rule out of the comparison.
        budget_range: what a campaign's budget is clamped to, and what the planner is told it
            may ask for. Set it from the run's size: it decides how many language-model calls
            a run costs, since one campaign is one call.
        token_noise: Gaussian noise on the input a token carries, in units of the problem's
            per-coordinate ``input_scale``. v2's continuous channel adds 0.05 on a unit range,
            about 0.17 of a coordinate's spread. The agent's own model keeps the true input.
        skip_peer_inputs: leave out of the candidate pool anything a neighbour's token shows was
            already run. Without it agents re-run each other's experiments: the exclusion covers
            only their own history and the round's other picks.
        skip_radius: how close a candidate must be to a reported input to be skipped. ``0``
            means the same input; with ``token_noise`` the reported input has moved, so this
            should cover the noise.
        hit_threshold: the score that makes an outcome a hit, for the hit tests and what the
            planner is told. ``None`` leaves hits undefined.
        share_hits: verdicts carry what the campaign found (hits, new hit families, best score)
            as well as whether its own test passed.
    """

    weights: ReasoningWeights = field(default_factory=ReasoningWeights)
    q_bits: int = 8
    memory_rounds: int = 3
    trust: bool = True
    trust_eta: float = 0.1
    warmup_rounds: int = 3
    pool_size: int = 256
    claim_weight: float = 1.0
    novelty_weight: float = 0.0
    share_descriptor: bool = True
    share_family: bool = True
    claims_on: bool = True
    share_text: bool = False
    claim_ttl: int = CLAIM_TTL
    verdict_ttl: int = VERDICT_TTL
    territory: bool = True
    territory_half_width: float = 0.25
    default_budget: int = 5
    budget_range: Tuple[int, int] = BUDGET_RANGE
    force_rule: Optional[str] = None
    token_noise: float = 0.0
    skip_peer_inputs: bool = False
    skip_radius: float = 0.0
    hit_threshold: Optional[float] = None
    share_hits: bool = False


class ADKOHAgent:
    """One lab. Owns a private GP, a history, a campaign, and a view of its neighbours.

    Args:
        agent_id: APPFL node id, what the transport routes by.
        index: federation index, what memory, trust and claims key on.
        surrogate: this agent's private model.
        problem: shared, read-only; it draws pools and runs experiments.
        cfg: the arm's configuration.
        neighbor_indices: federation indices of in-neighbours, for trust.
        degree: out-degree, which sets the FIFO budget with ``memory_rounds``.
        planner: ``None`` for a no-LLM arm, a campaign planner otherwise.
        centers: the run's territory seeds in the unit cube; ``None`` disables territories.
        descriptor_scale: the run's frozen per-component descriptor scale, for quantization.
    """

    def __init__(
        self,
        agent_id: str,
        index: int,
        surrogate: Surrogate,
        problem: Problem,
        cfg: Optional[AgentConfig] = None,
        *,
        neighbor_indices: Sequence[int] = (),
        degree: int = 1,
        seed: int = 0,
        planner: Optional[Any] = None,
        centers: Optional[np.ndarray] = None,
        descriptor_scale: Optional[Sequence[float]] = None,
    ):
        self.agent_id = str(agent_id)
        self.index = int(index)
        self.surrogate = surrogate
        self.problem = problem
        self.cfg = cfg or AgentConfig()
        self.space = problem.region_space
        self.planner = planner
        self.quantizer = Quantizer(self.cfg.q_bits)
        self.memory = TokenMemory(
            self.cfg.memory_rounds * max(int(degree), 1),
            self.cfg.weights.kernel,
            self.cfg.weights.bandwidth,
        )
        self.trust = TrustState(neighbor_indices, eta=self.cfg.trust_eta) if self.cfg.trust else None
        self.claims = ClaimStore(self.space, self.index)
        self.seed = int(seed)
        self.rng = np.random.default_rng([self.seed, self.index, 2025])
        self.token_rng = np.random.default_rng([self.seed, self.index, 2026])
        self.centers = None if centers is None else np.atleast_2d(np.asarray(centers, dtype=float))
        self.descriptor_scale = (
            None if descriptor_scale is None else np.asarray(descriptor_scale, dtype=float)
        )

        self.campaign: Optional[Campaign] = None
        self.campaigns: List[Campaign] = []
        self._pending: Optional[Dict[str, Any]] = None
        self._cands: Optional[Candidates] = None
        self.sigma_ref: Optional[float] = None

        # Own history. ``_u`` is the objective in maximize orientation, as v2 keeps it.
        self._keys: set = set()
        self._keylist: List[int] = []
        self._X: List[np.ndarray] = []
        self._u: List[float] = []
        self._desc: List[np.ndarray] = []
        self._fams: List[str] = []
        # What peers' tokens told me. The FIFO memory bounds how much a peer's *signal* can
        # pull the score -- that is a method choice the v2 findings settled. It must not also
        # decide how much of the team's footprint the novelty rule can see: forgetting that a
        # neighbour visited a point makes "furthest from everything anyone tried" mean
        # "furthest from what I tried recently", and ten agents rediscover the same thing.
        # These accumulate, and cost no communication -- the tokens already arrived.
        self._peer_X: List[np.ndarray] = []
        self._peer_desc: List[np.ndarray] = []
        self._peer_fams: set = set()
        self.diag: Dict[str, Any] = {}
        self.records: List[Dict[str, Any]] = []
        #: History entries that came from shared initial data rather than from this run's
        #: budget. Endpoints that count effort must skip them, or every agent holding the same
        #: initial points looks like the federation repeating itself.
        self._n_loaded = 0

    # -- shared initial data ---------------------------------------------------------------

    def load(self, cands: Candidates, outcomes: Sequence[Outcome]) -> None:
        """Take the shared initial points into the GP without spending budget."""
        for i, out in enumerate(outcomes):
            key, X, _F, _meta = cands.row(i)
            self._remember(key, X, out)
            self.surrogate.update(X, self._u[-1])
        self._n_loaded = len(self._u)

    # -- planning --------------------------------------------------------------------------

    def needs_plan(self, rnd: int) -> bool:
        """Whether the driver should ask the planner for a design before this round."""
        if self.planner is None or rnd < self.cfg.warmup_rounds:
            return False
        return self.campaign is None or self.campaign.done

    def territory_region(self):
        """The agent's own box, or the whole space when territories are off."""
        if not self.cfg.territory or self.centers is None or self.index >= len(self.centers):
            return self.space.full()
        return self.space.box_around(self.centers[self.index], self.cfg.territory_half_width)

    def plan_context(self, rnd: int) -> Dict[str, Any]:
        """What the planner sees. Its own data, its neighbours' claims, settled verdicts.

        No raw peer outcomes appear here -- an agent cannot see them, which is the constraint
        the whole method is built around, and the prompt must not quietly assume otherwise.
        """
        desc = self.problem.describe()
        order = [int(i) for i in np.argsort(self._u)[::-1][:8]] if self._u else []
        best = [
            {
                "point": self.problem.label(self._keylist[i], None),
                "score": round(float(self._u[i]), 4),
                "family": self._fams[i],
            }
            for i in order
        ]
        return {
            "agent": self.index,
            "round": rnd,
            "problem": desc,
            "dims": desc.get("dims", self.space.describe()),
            "rules": list(hrules.RULES),
            "budget_range": list(self.cfg.budget_range),
            "metrics": ["new_families", "novel_distance", "hull_growth", "max_objective"],
            "n_evaluated": len(self._u),
            "best": best,
            "my_families": sorted({f for f in self._fams if f})[:20],
            "peer_families": sorted(self._peer_fams)[:20],
            "territory": self.territory_region().label(self.space),
            "first_campaign": not self.campaigns,
            "active_claims": self.claims.claim_digest(rnd),
            "verdicts": self.claims.digest(),
            "history": [c.to_dict(self.space) for c in self.campaigns[-3:]],
            # What a planner needs to judge regions by the endpoint rather than by its own
            # test: what each recent campaign found, and the rates its tests should be set
            # against. Prompts that do not use these ignore them.
            "hit_threshold": self.cfg.hit_threshold,
            "claims_on": self.cfg.claims_on,
            "n_families": len({f for f in self._fams if f}),
            "n_peer_families": len(self._peer_fams),
            "base_rates": self._base_rates(),
            "recent": [
                {**c.to_dict(self.space), "found": self._found(c)} for c in self.campaigns[-6:]
            ],
        }

    def _is_hit(self, score: float) -> bool:
        t = self.cfg.hit_threshold
        if t is None:
            return False
        return score >= t if self.problem.maximize else score <= t

    def _found(self, c: Campaign) -> Dict[str, Any]:
        """What a campaign found, measured the way the endpoint is, against its own baseline."""
        known = c.baseline.get("known_families", set())
        scores = [float(r["score"]) for r in c.runs]
        fams = {r["family"] for r in c.runs if r["family"]}
        hit_fams = {r["family"] for r in c.runs if r["family"] and self._is_hit(float(r["score"]))}
        best = (max(scores) if self.problem.maximize else min(scores)) if scores else None
        return {
            "spent": len(c.runs),
            "hits": int(sum(self._is_hit(s) for s in scores)),
            "new_hit_families": len(hit_fams - known),
            "new_families": len(fams - known),
            "best": None if best is None else round(best, 4),
            "mean": round(float(np.mean(scores)), 4) if scores else None,
        }

    def _base_rates(self) -> Optional[Dict[str, float]]:
        """Per-experiment rates over this agent's finished campaigns: what a test should beat."""
        done = [self._found(c) for c in self.campaigns if c.verdict is not None]
        n = sum(f["spent"] for f in done)
        if not n:
            return None
        return {
            "campaigns": len(done),
            "experiments": n,
            "new_families": sum(f["new_families"] for f in done) / n,
            "hits": sum(f["hits"] for f in done) / n,
            "new_hit_families": sum(f["new_hit_families"] for f in done) / n,
        }

    def set_campaign(self, design: Optional[Dict[str, Any]], rnd: int) -> Campaign:
        """Open a campaign from a design, snapshot its baseline, and announce its claim.

        ``design`` of ``None`` means no planner: one open-ended campaign over the whole space
        with the settled rule, which is exactly ADKO v2 with a pool instead of a grid.
        """
        if design is None:
            region = self.space.full()
            c = Campaign(
                region=region,
                rule=hrules.valid_rule(self.cfg.force_rule or "adko"),
                budget=10**9,
                test=Test("new_families", ">=", 1.0),
                claim_id=f"a{self.index}c0",
                started=rnd,
                source="none",
                budget_range=self.cfg.budget_range,
            )
        else:
            region = design.get("region") or self.space.full()
            if self.cfg.territory and not self.campaigns:
                region = self._intersect(region, self.territory_region())
            c = Campaign(
                region=region,
                rule=hrules.valid_rule(self.cfg.force_rule or design.get("rule", "adko")),
                budget=int(design.get("budget", self.cfg.default_budget)),
                hypothesis=str(design.get("hypothesis", ""))[:300],
                rationale=str(design.get("rationale", ""))[:600],
                test=design.get("test") or Test(),
                p_success=float(design.get("p_success", 0.5)),
                claim_id=f"a{self.index}c{len(self.campaigns)}",
                started=rnd,
                source=str(design.get("source", "llm")),
                budget_range=self.cfg.budget_range,
            )
        c.baseline = self._snapshot()
        self.campaign = c
        self.campaigns.append(c)
        if self.cfg.claims_on and design is not None:
            self.claims.announce(
                Claim(
                    origin=self.index,
                    claim_id=c.claim_id,
                    region=c.region.pack(self.space),
                    rule=c.rule,
                    hypothesis=c.hypothesis[:CLAIM_CHARS] if self.cfg.share_text else "",
                    round=rnd,
                    expires=rnd + c.budget + CLAIM_SLACK,
                    ttl=self.cfg.claim_ttl,
                )
            )
        return c

    def _intersect(self, a, b):
        """The tighter of two regions, coordinate by coordinate."""
        bounds = dict(getattr(b, "bounds", {}))
        for name, (lo, hi) in getattr(a, "bounds", {}).items():
            if name in bounds:
                blo, bhi = bounds[name]
                lo, hi = max(lo, blo), min(hi, bhi)
                if hi <= lo:
                    lo, hi = blo, bhi
            bounds[name] = (lo, hi)
        picks = dict(getattr(b, "picks", {}))
        for name, opts in getattr(a, "picks", {}).items():
            keep = tuple(sorted(set(opts) & set(picks[name]))) if name in picks else tuple(opts)
            picks[name] = keep or picks.get(name, tuple(opts))
        return self.space.region(bounds=bounds, picks=picks)

    # -- one experiment --------------------------------------------------------------------

    def propose(self, rnd: int, taken: Sequence[int] = ()) -> Pick:
        """This round's experiment: the rule's argmax inside the campaign's region.

        ``taken`` are keys already claimed by another agent this round. Nothing is evaluated
        twice in a round, as the reference comparisons require, and the exclusion happens in
        the pool draw so the rule never sees a point it cannot have.
        """
        warm = rnd < self.cfg.warmup_rounds
        if not warm and self.campaign is None:
            self.set_campaign(
                None
                if self.planner is None
                else random_design(self.space, self.rng, budget_range=self.cfg.budget_range),
                rnd,
            )
        c = self.campaign
        region = self.territory_region() if warm or c is None else c.region
        exclude = sorted(self._keys | {int(k) for k in taken})
        cands = self.problem.pool(self.rng, self.cfg.pool_size, exclude, region)
        skipped = 0
        if self.cfg.skip_peer_inputs and self._peer_X:
            cands, skipped = self._skip_reported(cands, exclude, region)
        self._cands = cands
        if len(cands) == 0:
            raise RuntimeError(f"{self.problem.problem_id}: empty candidate pool at round {rnd}")
        mask = region.mask(self.space, cands.F)

        if warm or c is None:
            idx = np.flatnonzero(mask)
            i = int(self.rng.choice(idx if idx.size else np.arange(len(cands))))
            self.diag = {"mode": "warmup", "rule": "random", "pool": len(cands)}
            rule, claim_id = "random", ""
        else:
            i = self._score_pick(cands, mask, rnd, c)
            rule, claim_id = c.rule, c.claim_id
        if self.cfg.skip_peer_inputs:
            self.diag["skipped"] = skipped
        key, X, F, meta = cands.row(i)
        return Pick(agent=self.index, key=key, X=X, F=F, meta=meta, rule=rule, claim_id=claim_id)

    def _skip_reported(self, cands: Candidates, exclude: Sequence[int], region) -> Tuple[Candidates, int]:
        """Drop candidates a neighbour's token shows were already run; top the pool back up.

        Tokens carry inputs, not keys, so a candidate counts as reported when it lies within
        ``skip_radius`` of a reported input (``0``: the same input). Refills come from the same
        region with everything drawn so far excluded, at most three times, so a region that
        neighbours have mined out leaves a smaller pool rather than a stall -- and if nothing at
        all is left, the original draw stands.
        """
        known = np.asarray(self._peer_X, dtype=float)
        tol = max(float(self.cfg.skip_radius), 1e-9)
        ok = hrules.nearest_distance(cands.X, known) > tol
        skipped = int(np.count_nonzero(~ok))
        if not skipped:
            return cands, 0
        parts, have = [_take(cands, ok)], int(np.count_nonzero(ok))
        drawn = set(int(k) for k in exclude) | {int(k) for k in cands.key}
        for _ in range(3):
            if have >= self.cfg.pool_size:
                break
            more = self.problem.pool(self.rng, self.cfg.pool_size - have, sorted(drawn), region)
            if len(more) == 0:
                break
            drawn |= {int(k) for k in more.key}
            ok = hrules.nearest_distance(more.X, known) > tol
            skipped += int(np.count_nonzero(~ok))
            parts.append(_take(more, ok))
            have += int(np.count_nonzero(ok))
        out = _join(parts)
        return (out if len(out) else cands), skipped

    def _score_pick(self, cands: Candidates, mask: np.ndarray, rnd: int, c: Campaign) -> int:
        w = self.cfg.weights
        mu, sigma = self.surrogate.predict(cands.X)
        G0, P0 = PeerField(self.memory, self.source_weights())(cands.X)
        cal = Calibrator(sigma, G0, P0, w.eps)
        G, L = cal.apply(G0, P0)
        self.sigma_ref = float(sigma.max())
        nov = hrules.nearest_distance(cands.X, self._known_inputs())
        v = hrules.values(c.rule, mu, sigma, G, L, nov, beta=w.beta, lam=w.lam, gamma=w.gamma)
        pen = self.claims.penalty(cands.F, rnd) if self.cfg.claims_on else None
        v, parts = hrules.adjust(
            v,
            sigma,
            penalty=pen,
            penalty_weight=self.cfg.claim_weight,
            novelty=nov,
            novelty_weight=self.cfg.novelty_weight,
        )
        i = hrules.pick(v, mask)
        self.diag = {
            "mode": "campaign",
            "rule": c.rule,
            "pool": len(cands),
            "in_region": int(np.count_nonzero(mask)),
            "region_empty": not bool(mask.any()),
            "claim_id": c.claim_id,
            "spent": c.spent,
            "budget": c.budget,
            "steered": float(np.mean(pen)) if pen is not None else 0.0,
            "score_baseline": bool(
                i == hrules.pick(score(mu, sigma, G, L, w.beta, w.lam, w.gamma), None)
            ),
            **{f"term_{k}": round(float(x), 6) for k, x in parts.items()},
            **{k: cal.diagnostics[k] for k in ("k", "U", "floor_active")},
        }
        return i

    def complete(self, pick: Pick, out: Outcome, rnd: int) -> HToken:
        """Record the result, close the campaign if this was its last experiment, emit a token."""
        c = self.campaign
        last = c.last if c is not None else False
        self._remember(pick.key, pick.X, out)
        self.surrogate.update(pick.X, self._u[-1])
        verdicts: List[Verdict] = []
        insight = None
        if c is not None:
            c.runs.append(
                {
                    "round": rnd,
                    "key": int(pick.key),
                    "point": self.problem.label(int(pick.key), pick.meta),
                    "score": round(float(out.score), 6),
                    "family": out.family,
                }
            )
            if last or c.done:
                v = c.judge(self._evidence(c))
                insight = self._insight(c, v)
                if self.cfg.claims_on:
                    found = self._found(c) if self.cfg.share_hits else None
                    verdict = Verdict(
                        origin=self.index,
                        claim_id=c.claim_id,
                        region=c.region.pack(self.space),
                        metric=v["metric"],
                        value=v["value"],
                        op=v["op"],
                        threshold=v["threshold"],
                        success=v["success"],
                        hypothesis=c.hypothesis[: VERDICT_CHARS if self.cfg.share_text else 80],
                        round=rnd,
                        ttl=self.cfg.verdict_ttl,
                        **(
                            {}
                            if found is None
                            else {
                                "spent": found["spent"],
                                "hits": found["hits"],
                                "hit_families": found["new_hit_families"],
                                "best": found["best"],
                            }
                        ),
                    )
                    self.claims.settle(verdict)
                if self.planner is not None:
                    getattr(self.planner, "record_outcome", lambda *a, **k: None)(c, v)
        claims_out, verdicts_out = self.claims.outbox() if self.cfg.claims_on else ([], [])
        x_shared = self.problem.privatize(pick.X, self.token_rng)
        if self.cfg.token_noise > 0:
            x_shared = self._jitter(x_shared)
        tok = encode(
            agent_id=self.agent_id,
            source=self.index,
            round=rnd,
            x_shared=x_shared,
            u=self._u[-1],
            history=self._u,
            quantizer=self.quantizer,
            new_best=len(self._u) == 1 or self._u[-1] > max(self._u[:-1]),
            insight=insight,
            descriptor=self._shared_descriptor(out.descriptor),
            family=out.family if self.cfg.share_family else "",
            claims=claims_out,
            verdicts=verdicts_out,
        )
        self.records.append(
            {
                "round": rnd,
                "agent": self.index,
                "key": int(pick.key),
                "point": self.problem.label(int(pick.key), pick.meta),
                "score": float(out.score),
                "family": out.family,
                "descriptor": [float(z) for z in np.asarray(out.descriptor, dtype=float).ravel()],
                "rule": pick.rule,
                "claim_id": pick.claim_id,
                "bits": tok.payload_bits(),
                "trust": None if self.trust is None else self.trust.mean(),
                **{f"d_{k}": x for k, x in self.diag.items()},
                **({"extras": out.extras} if out.extras else {}),
            }
        )
        return tok

    def _insight(self, c: Campaign, v: Dict[str, Any]) -> str:
        """The one line that leaves with the campaign's last result: claim, reason, outcome."""
        val = "n/a" if v["value"] is None else f"{v['value']:g}"
        return (
            f"{c.hypothesis or '(no hypothesis)'} | why: {c.rationale or '(none)'} | "
            f"{'held' if v['success'] else 'failed'} ({v['metric']}={val} vs {v['op']} {v['threshold']:g})"
        )[:400]

    # -- bookkeeping -----------------------------------------------------------------------

    def _remember(self, key: int, X: np.ndarray, out: Outcome) -> None:
        self._keys.add(int(key))
        self._keylist.append(int(key))
        self._X.append(np.asarray(X, dtype=float))
        self._u.append(float(out.score) if self.problem.maximize else -float(out.score))
        self._desc.append(np.asarray(out.descriptor, dtype=float).ravel())
        self._fams.append(str(out.family or ""))

    @property
    def own_picks(self) -> List[np.ndarray]:
        """The embeddings this agent actually spent budget on."""
        return self._X[self._n_loaded :]

    def _known_inputs(self) -> np.ndarray:
        """Everything anyone has run, as far as I know: my points plus every peer input I ever
        received -- not just the ones still in the FIFO."""
        rows = list(self._X) + list(self._peer_X)
        if not rows:
            return np.zeros((0, 1))
        d = max(len(x) for x in rows)
        return np.asarray([x for x in rows if len(x) == d], dtype=float)

    def _descriptors(self, include_peers: bool = True) -> np.ndarray:
        rows = [z for z in self._desc if z.size]
        if include_peers:
            rows += [z for z in self._peer_desc if z.size]
        if not rows:
            return np.zeros((0, 0))
        d = max(len(z) for z in rows)
        return np.asarray([z for z in rows if len(z) == d], dtype=float)

    def _snapshot(self) -> Dict[str, Any]:
        Z = self._descriptors()
        return {
            "known_descriptors": Z,
            "known_families": {f for f in self._fams if f} | set(self._peer_fams),
            "hull": hmetrics.hull_volume(Z),
            "n": len(self._u),
        }

    def _evidence(self, c: Campaign) -> Evidence:
        """The campaign's own runs against the snapshot taken when it opened."""
        n = len(c.runs)
        mine = self._descriptors(include_peers=False)
        Z = mine[-n:] if n and len(mine) >= n else np.zeros((0, 0))
        base = c.baseline.get("known_descriptors", np.zeros((0, 0)))
        after = np.vstack([base, Z]) if base.size and Z.size and base.shape[1] == Z.shape[1] else (
            Z if Z.size else base
        )
        return Evidence(
            scores=np.asarray([r["score"] for r in c.runs], dtype=float),
            descriptors=Z,
            families=[r["family"] for r in c.runs],
            known_descriptors=base,
            known_families=set(c.baseline.get("known_families", set())),
            hull_before=float(c.baseline.get("hull", 0.0)),
            hull_after=hmetrics.hull_volume(after),
            hit_threshold=self.cfg.hit_threshold,
            maximize=self.problem.maximize,
        )

    def _jitter(self, x) -> np.ndarray:
        """v2's continuous input channel: Gaussian noise on the shared input, per coordinate."""
        x = np.asarray(x, dtype=float)
        scale = self.problem.input_scale()
        s = np.ones(x.shape) if scale is None else np.broadcast_to(np.asarray(scale, dtype=float), x.shape)
        return x + self.token_rng.normal(0.0, 1.0, size=x.shape) * (self.cfg.token_noise * s)

    def _shared_descriptor(self, z: np.ndarray) -> List[float]:
        if not self.cfg.share_descriptor or self.descriptor_scale is None:
            return []
        return quantize_descriptor(np.asarray(z, dtype=float).ravel(), self.descriptor_scale)

    # -- neighbours ------------------------------------------------------------------------

    def observe(self, tokens: Sequence[HToken]) -> None:
        """Ingest neighbour tokens: trust, memory, footprints, claims and verdicts."""
        peers = sorted(
            [t for t in tokens if t.source != self.index],
            key=lambda t: (t.round, t.source),
        )
        if not peers:
            return
        if self.trust is not None:
            for t in peers:
                self.trust.ensure(t.source)
            if self.surrogate.fitted:
                self._update_trust(peers)
        self.memory.add(peers)
        self.memory.prune()
        for t in peers:
            self._peer_X.append(np.asarray(t.x, dtype=float))
            z = np.asarray(getattr(t, "descriptor", []), dtype=float)
            if z.size and self.descriptor_scale is not None and z.size == len(self.descriptor_scale):
                self._peer_desc.append(z * self.descriptor_scale)
            fam = getattr(t, "family", "")
            if fam:
                self._peer_fams.add(fam)
        if self.cfg.claims_on:
            self.claims.ingest(
                [c for t in peers for c in getattr(t, "claims", [])],
                [v for t in peers for v in getattr(t, "verdicts", [])],
            )

    def _update_trust(self, tokens: Sequence[HToken]) -> None:
        mu, sd = self.surrogate.predict(np.asarray([t.x for t in tokens], dtype=float))
        qhat = [
            score_against(float(m), self._u, self.quantizer, maximize=True)
            for m in self.surrogate.raw_mean(mu)
        ]
        rel = relative_sigma(sd, self.sigma_ref)
        by_src: Dict[int, List[float]] = defaultdict(list)
        for t, qh, s in zip(tokens, qhat, rel):
            by_src[t.source].append(agreement(t.q, qh, s))
        for j, a in by_src.items():
            self.trust.update(j, a)

    def source_weights(self) -> Optional[Dict[int, float]]:
        return None if self.trust is None else self.trust.weights()

    # -- read-outs -------------------------------------------------------------------------

    def best_so_far(self) -> Optional[Tuple[int, float]]:
        if not self._u:
            return None
        i = int(np.argmax(self._u))
        return (self._keylist[i], float(self._u[i]))

    def trace(self) -> Dict[str, Any]:
        """Everything the run record keeps about this agent."""
        judged = [c for c in self.campaigns if c.verdict]
        return {
            "agent": self.index,
            "agent_id": self.agent_id,
            "n_evaluated": len(self._u),
            "best": max(self._u) if self._u else None,
            "families": sorted({f for f in self._fams if f}),
            "peer_families": sorted(self._peer_fams),
            "campaigns": [c.to_dict(self.space) for c in self.campaigns],
            "calibration": hmetrics.calibration(
                [c.p_success for c in judged], [bool(c.verdict["success"]) for c in judged]
            ),
            "claims": dict(self.claims.totals),
            "trust": None if self.trust is None else self.trust.snapshot(),
            "records": self.records,
        }


def _take(c: Candidates, mask: np.ndarray) -> Candidates:
    """The rows of a pool where ``mask`` holds."""
    idx = np.flatnonzero(mask)
    meta = list(c.meta)
    return Candidates(
        key=np.asarray(c.key)[idx],
        X=np.asarray(c.X)[idx],
        F=np.asarray(c.F)[idx],
        meta=[meta[i] for i in idx] if meta else [],
    )


def _join(parts: Sequence[Candidates]) -> Candidates:
    """Several pools as one, in order."""
    return Candidates(
        key=np.concatenate([np.asarray(p.key) for p in parts]),
        X=np.vstack([np.asarray(p.X) for p in parts]),
        F=np.vstack([np.asarray(p.F) for p in parts]),
        meta=[m for p in parts for m in p.meta],
    )
