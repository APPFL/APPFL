"""The campaign planner: one batched call per round, one served model for every agent.

Every agent whose campaign has ended is planned in the same round, and those calls go out
together against one vLLM server -- the agents are separate contexts, not separate models,
which is what makes a single 27B model viable for ten agents now and plausible for a hundred
later.

Three things this does that the L3 caller did not, all for the same reason -- a run must be
reproducible and affordable at scale:

* **cache on the prompt.** A repeated design decision replays from SQLite for free, so a
  crashed run costs its GPU time again and not its tokens.
* **fail soft, and say so.** A call that errors, times out or returns something unusable
  leaves the agent with a random design and increments an error counter that lands in the run
  record. A run that silently degraded to the control arm would be worse than a crash.
* **log every call.** Prompt hash, usage, latency, the parsed design and the raw text go to
  ``llm_calls.jsonl``, which is what the screen's reports are built from.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from appfl.decentralized.algorithm.adko_h.campaign import random_design
from appfl.decentralized.algorithm.adko_h.llm import prompts, schema
from appfl.decentralized.algorithm.adko_v2.llm.client import Cache, request_key

#: Default serving settings for an open model behind vLLM.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_MAX_TOKENS = 4096


class CampaignPolicy:
    """Plans campaigns with a served model. Duck-typed to what ``loop.run`` calls.

    Args:
        client: anything with ``async complete(request, ctx)``; ``LocalClient`` for vLLM.
        space: the region space, for the schema and for parsing.
        run_dir: where the call log and cache go.
        model: served model name.
        seed: used for the fallback designs, so a failed call is still reproducible.
        temperature: 0.2 keeps designs stable between arms; higher spreads them. ``None``
            sends none, which is what a provider that fixes it (a thinking model) needs.
        thinking: ``"low"``, ``"medium"``, ``"high"`` or ``None`` to leave the server's default.
            vLLM only; on MAG, effort is part of the model id.
        max_error_rate: a run whose planner fails more often than this (after ten designs)
            stops. Below it, a failed call becomes a logged random design; above it, the arm
            would be the random control under another name.
        max_tokens: completion cap, reasoning included.
        prompt: ``"v1"`` (the first screen's prompt) or ``"v2"`` (hits, findings, base rates;
            see ``prompts.py``). The version also sets which test metrics the schema allows.
        note: ask for the sentence the detailed channel carries to the neighbours. Set from
            ``AgentConfig.share_detail``, so the schema and the token agree.
    """

    def __init__(
        self,
        client: Any,
        space,
        run_dir: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        seed: int = 0,
        temperature: Optional[float] = 0.2,
        thinking: Optional[str] = "low",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_cache: bool = True,
        budget_range=None,
        max_error_rate: float = 0.2,
        prompt: str = "v1",
        note: bool = False,
    ):
        if prompt not in ("v1", "v2"):
            raise ValueError(f"prompt must be 'v1' or 'v2', got {prompt!r}")
        self.client = client
        self.space = space
        self.model = str(model)
        self.temperature = None if temperature is None else float(temperature)
        self.max_error_rate = float(max_error_rate)
        self.thinking = thinking
        self.max_tokens = int(max_tokens)
        self.budget_range = tuple(budget_range) if budget_range else None
        self.prompt = prompt
        self.format = schema.campaign_format(
            space,
            metrics=schema.METRICS_V2 if prompt == "v2" else schema.METRICS,
            budget_range=self.budget_range,
            note=bool(note) and prompt == "v2",
        )
        self.rng = np.random.default_rng([int(seed), 31337])
        self.dir = Path(run_dir) if run_dir else None
        self.cache = None
        self.log = None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.log = (self.dir / "llm_calls.jsonl").open("a")
            if use_cache:
                self.cache = Cache(str(self.dir / "llm_cache.sqlite"))
        self.totals: Dict[str, float] = {
            "calls": 0,
            "designs": 0,
            "cached": 0,
            "errors": 0,
            "unusable": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "seconds": 0.0,
        }
        #: Campaign verdicts, kept so a report can score the planner without the run record.
        self.outcomes: List[Dict[str, Any]] = []
        # One event loop for the planner's whole life. An async HTTP client binds its
        # connection pool to the loop it first ran on, so asyncio.run() per round -- a fresh
        # loop each time, closed after -- fails every call after the first round with "Event
        # loop is closed". The mock client cannot show this; the first real gateway run did.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _run(self, coro):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    # -- what the loop calls ----------------------------------------------------------------

    def plan(self, contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One design per context, asked for concurrently."""
        if not contexts:
            return []
        self.totals["calls"] += 1
        self.totals["designs"] += len(contexts)
        return self._run(self._plan(list(contexts)))

    async def _plan(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = await asyncio.gather(*[self._one(c) for c in contexts], return_exceptions=True)
        for res in out:
            if type(res).__name__ == "BudgetSpent":
                raise res  # out of money: stop the run, do not become the control arm
        designs: List[Dict[str, Any]] = []
        for ctx, res in zip(contexts, out):
            if isinstance(res, BaseException):
                self.totals["errors"] += 1
                self._write({"agent": ctx.get("agent"), "round": ctx.get("round"), "error": repr(res)})
                designs.append(self._fallback("error"))
            else:
                designs.append(res)
        bad = self.totals["errors"] + self.totals["unusable"]
        if self.totals["designs"] >= 10 and bad > self.max_error_rate * self.totals["designs"]:
            raise RuntimeError(
                f"planner failed {int(bad)} of {int(self.totals['designs'])} designs "
                f"(limit {self.max_error_rate:.0%}); see llm_calls.jsonl"
            )
        return designs

    async def _one(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        request = self._request(ctx)
        key = request_key(request, getattr(self.client, "name", "local"))
        t0 = time.time()
        hit = self.cache.get(key) if self.cache else None
        if hit is not None:
            self.totals["cached"] += 1
            resp = hit
        else:
            resp = await self.client.complete(request, ctx)
            if self.cache:
                self.cache.put(key, resp)
        secs = time.time() - t0
        usage = resp.get("usage") or {}
        self.totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        self.totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        self.totals["reasoning_tokens"] += int(usage.get("reasoning_tokens", 0) or 0)
        if hit is None:  # a replayed call costs nothing
            self.totals["cost_usd"] += float(usage.get("cost_usd", 0) or 0)
        self.totals["seconds"] += secs
        design = schema.parse(self.space, _json(resp.get("text", "")))
        if design is None:
            self.totals["unusable"] += 1
            design = self._fallback("unusable")
        self._write(
            {
                "agent": ctx.get("agent"),
                "round": ctx.get("round"),
                "key": key,
                "cached": hit is not None,
                "seconds": round(secs, 3),
                "usage": usage,
                "status": resp.get("status"),
                "design": {
                    "region": design["region"].label(self.space),
                    "rule": design["rule"],
                    "budget": design["budget"],
                    "test": design["test"].label(),
                    "p_success": design["p_success"],
                    "hypothesis": design["hypothesis"],
                    "rationale": design["rationale"],
                    "source": design["source"],
                },
                "text": (resp.get("text") or "")[:2000],
                "reasoning": (resp.get("reasoning") or "")[:2000],
            }
        )
        return design

    def record_outcome(self, campaign, verdict: Dict[str, Any]) -> None:
        """A closed campaign's verdict, for the report and for the planner's own record."""
        self.outcomes.append(
            {
                "claim_id": campaign.claim_id,
                "region": campaign.region.label(self.space),
                "rule": campaign.rule,
                "budget": campaign.budget,
                "hypothesis": campaign.hypothesis,
                **{k: verdict.get(k) for k in ("metric", "value", "op", "threshold", "success")},
                "p_success": campaign.p_success,
            }
        )

    def close(self) -> None:
        if self.cache:
            self.cache.close()
        if self.log:
            self.log.close()
            self.log = None
        closer = getattr(self.client, "close", None)
        if closer is not None:
            try:
                self._run(closer())
            except (RuntimeError, TypeError):
                pass
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()

    # -- internals --------------------------------------------------------------------------

    def _request(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        v2 = self.prompt == "v2"
        req = {
            "model": self.model,
            "instructions": prompts.system_v2(ctx) if v2 else prompts.SYSTEM,
            "input": prompts.user_v2(ctx) if v2 else prompts.user(ctx),
            "reasoning": {"effort": self.thinking or "low", "summary": "auto"},
            "text": {"format": self.format},
            "store": False,
            "max_output_tokens": self.max_tokens,
        }
        extra: Dict[str, Any] = {}
        if self.temperature is not None:
            extra["temperature"] = self.temperature
        if self.thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": True}
            extra["reasoning_effort"] = self.thinking
        req["extra"] = extra
        return req

    def _fallback(self, why: str) -> Dict[str, Any]:
        d = random_design(self.space, self.rng, budget_range=self.budget_range or (2, 10))
        d["source"] = f"fallback:{why}"
        d["rationale"] = f"planner {why}; random design used instead"
        return d

    def _write(self, row: Dict[str, Any]) -> None:
        if self.log:
            self.log.write(json.dumps(row, default=str) + "\n")
            self.log.flush()


def _json(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a reply, tolerating fences and a thinking preamble."""
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        s = s[4:] if s.lower().startswith("json") else s
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            return json.loads(s[a : b + 1])
        except json.JSONDecodeError:
            return None


class MockClient:
    """Deterministic, offline designs that still vary with the context.

    Enough to exercise the schema, the parser, the claim channel and the verdicts without a
    GPU or a gateway, which is how the loop gets tested anywhere. Its designs are not meant to
    be good: an arm run against this mock should land near the random control, and if it does
    not, something in the machinery is leaking information.
    """

    name = "mock"

    def __init__(self, space, seed: int = 0):
        self.space = space
        self.seed = int(seed)
        self.calls = 0

    async def complete(self, request: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None):
        self.calls += 1
        ctx = ctx or {}
        rng = np.random.default_rng([self.seed, int(ctx.get("agent", 0)), int(ctx.get("round", 0))])
        region = self.space.box_around(rng.random(len(self.space)), float(rng.uniform(0.1, 0.35)))
        body = {
            "region": {
                **{
                    d.name: (
                        [float(v) for v in region.bounds[d.name]] if d.name in region.bounds else None
                    )
                    for d in self.space.dims
                    if d.kind == "real"
                },
                **{
                    d.name: ([int(i) for i in region.picks[d.name]] if d.name in region.picks else None)
                    for d in self.space.dims
                    if d.kind == "cat"
                },
            },
            "rule": str(rng.choice(["adko", "novelty", "explore", "indep", "exploit"])),
            "budget": int(rng.integers(3, 8)),
            "hypothesis": f"mock: unexplored corner near {region.label(self.space)}",
            "rationale": "mock client; no reasoning performed",
            "test": {
                "metric": str(rng.choice(["new_families", "novel_distance"])),
                "op": ">=",
                "threshold": float(rng.integers(1, 3)),
            },
            "p_success": float(round(rng.uniform(0.3, 0.8), 2)),
        }
        return {
            "text": json.dumps(body),
            "summary": None,
            "reasoning": "",
            "usage": {"input_tokens": 1400, "output_tokens": 220, "reasoning_tokens": 0},
            "response_id": f"mock-{self.calls}",
            "status": "stop",
        }

    async def close(self) -> None:
        return None


def build_planner(
    space,
    kind: str = "mock",
    *,
    base_url: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    run_dir: Optional[str] = None,
    seed: int = 0,
    concurrency: int = 16,
    **kwargs,
) -> Optional[CampaignPolicy]:
    """``kind``: ``"none"`` returns ``None``; ``"mock"`` runs offline; ``"vllm"`` needs a URL;
    ``"mag"`` uses the AmSC gateway (see ``mag.py`` for its budget and rate-limit settings).

    Credentials come from the environment the client reads, never from a configuration file
    and never from a log line.
    """
    if kind in ("none", None):
        return None
    if kind == "mock":
        client: Any = MockClient(space, seed)
    elif kind in ("vllm", "local"):
        from appfl.decentralized.algorithm.adko_v2.llm.client import LocalClient

        if not base_url:
            raise ValueError("a vLLM planner needs base_url, e.g. http://host:8000/v1")
        client = LocalClient(base_url, concurrency=concurrency)
    elif kind == "mag":
        from appfl.decentralized.algorithm.adko_h.llm.mag import MagClient

        client = MagClient()
        kwargs["thinking"] = None  # effort is in the MAG model id, not a request field
    else:
        raise ValueError(f"unknown planner kind {kind!r}; use none, mock, vllm or mag")
    return CampaignPolicy(client, space, run_dir=run_dir, model=model, seed=seed, **kwargs)
