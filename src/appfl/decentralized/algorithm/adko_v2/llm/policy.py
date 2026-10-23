"""L3: with probability alpha an agent's LLM picks its next experiment;
the random control picks a random untried condition on the same draws."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from .client import (Cache, LocalClient, MockClient, OpenAIClient,
                     cost_usd, make_request, request_key)
from .prompts import render_system, render_user, shown_ids
from .schema import choose, hypothesis_of, parse, shared_text

TOTALS = ("calls", "cache_hits", "errors", "skipped", "input_tokens",
          "output_tokens", "reasoning_tokens", "cost_usd", "latency_s")


def coin(seed, agent, rnd, alpha, warmup):
    """u(seed, agent, round) < alpha: the same draws for every alpha and
    for the random control."""
    if rnd < warmup or alpha <= 0:
        return False
    rng = np.random.default_rng([int(seed), int(agent), int(rnd), 3])
    return bool(rng.random() < alpha)


class L3Policy:
    def __init__(self, client, run_dir, seed, setting, alpha=0.3,
                 model="gpt-6.1-sol", effort="high", max_usd=None,
                 n_agents=4, warmup=5, run_id=None, share="hypothesis",
                 solo=False, max_tokens=25000,
                 log_name="llm_calls.jsonl"):  # APPFL: one log per lab
        self.client, self.seed, self.setting = client, int(seed), setting
        self.alpha, self.model, self.effort = float(alpha), model, effort
        self.max_usd = None if max_usd is None else float(max_usd)
        self.n_agents, self.warmup = int(n_agents), int(warmup)
        self.share, self.solo = share, bool(solo)  # solo: no peer tokens
        self.max_tokens = int(max_tokens)  # per call, reasoning included
        self.run_id = run_id or f"L3-{setting}-s{seed}"
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(run_dir / "llm_cache.sqlite")
        self.log = open(run_dir / log_name, "a")
        self.loop = asyncio.new_event_loop()
        self.past, self.by_call = {}, {}
        self.totals = dict.fromkeys(TOTALS, 0)

    def wants(self, rnd, agent):
        return coin(self.seed, agent, rnd, self.alpha, self.warmup)

    def decide(self, contexts):
        if self.max_usd is not None and \
                self.totals["cost_usd"] >= self.max_usd:
            return [self._skip(c, "max_usd reached") for c in contexts]
        return self.loop.run_until_complete(self._batch(contexts))

    async def _batch(self, contexts):
        return await asyncio.gather(*(self._one(c) for c in contexts))

    async def _one(self, ctx):
        agent, rnd = int(ctx["agent"]), int(ctx["round"])
        call_id = f"{self.run_id}-a{agent}-r{rnd}"
        system = render_system(agent, self.n_agents, self.share)
        view = dict(ctx, memory=[]) if self.solo else ctx
        past = list(self.past.get(agent, []))  # as shown, before this call
        user = render_user(view, past, self.share)
        request = make_request(self.model, self.effort, system, user,
                               self.max_tokens)
        key = request_key(request, self.client.name)
        resp, error, start = self.cache.get(key), None, time.time()
        hit = resp is not None
        if not hit:
            try:
                resp = await self.client.complete(request, ctx)
                self.cache.put(key, resp)
            except Exception as exc:  # logged; the loop uses its score
                error = f"{type(exc).__name__}: {exc}"
        parsed = parse(resp["text"]) if resp else None
        if parsed:
            x, source, reason = choose(parsed, ctx)
        else:
            x, source, reason = None, "none", error or "unparseable"
        free = hit or not resp or getattr(self.client, "free", False)
        cost = 0.0 if free else cost_usd(self.model, resp["usage"])
        latency = round(time.time() - start, 3)
        self._count(resp, hit, error, cost, latency)
        out = resp or {}
        hyp = hypothesis_of(parsed) if x is not None else None
        insight = None if x is None else shared_text(
            parsed, out.get("summary") or out.get("reasoning"), self.share)
        if x is not None:
            entry = {"round": rnd, "x": x, "hypothesis": hyp, "y": None,
                     "call_id": call_id}
            self.past.setdefault(agent, []).append(entry)
            self.by_call[call_id] = entry
        shown = shown_ids(view)
        cited = [str(i) for i in (parsed or {}).get("evidence_ids") or []]
        self._write({
            "type": "call", **self._ids(ctx, call_id),
            "model": self.model, "effort": self.effort,
            "system": system, "user": user, "shown_token_ids": shown,
            "response": parsed,
            "raw_text": None if parsed or not resp else resp["text"],
            "reasoning_summary": out.get("summary"),
            "reasoning": out.get("reasoning"),  # full trace (local models)
            # provenance: the calls whose outcomes the prompt showed, and
            # the token this decision's experiment sends
            "past_call_ids": [d["call_id"] for d in past[-5:]],
            "token_id": f"a{agent}r{rnd}",
            "evidence_ids": [i for i in cited if i in shown],
            "evidence_unknown": [i for i in cited if i not in shown],
            "executed": {"x": x, "source": source, "reason": reason},
            "shared": insight,
            "usage": (resp or {}).get("usage"),
            "latency_s": latency,
            "response_id": (resp or {}).get("response_id"),
            "status": (resp or {}).get("status"), "cache_hit": hit,
            "cost_usd": cost, "error": error})
        return {"x": x, "source": source, "insight": insight,
                "call_id": call_id}

    def record_outcome(self, call_id, y, q):
        if call_id in self.by_call:
            self.by_call[call_id]["y"] = float(y)
        self._write({"type": "outcome", "call_id": call_id, "y": float(y),
                     "q": None if q is None else float(q)})

    def close(self):
        self._write({"type": "summary", "run": self.run_id,
                     "max_usd": self.max_usd, **self.totals})
        self.log.close()
        self.cache.close()
        self.loop.run_until_complete(self.client.close())
        self.loop.close()

    def _skip(self, ctx, reason):
        call_id = f"{self.run_id}-a{ctx['agent']}-r{ctx['round']}"
        self.totals["skipped"] += 1
        self._write({"type": "skip", **self._ids(ctx, call_id),
                     "reason": reason})
        return {"x": None, "source": "none", "insight": None,
                "call_id": call_id}

    def _ids(self, ctx, call_id):
        return {"run": self.run_id, "seed": self.seed,
                "setting": self.setting, "agent": int(ctx["agent"]),
                "round": int(ctx["round"]), "call_id": call_id}

    def _count(self, resp, hit, error, cost, latency):
        t = self.totals
        t["calls"] += 1
        t["cache_hits"] += int(hit)
        t["errors"] += int(error is not None)
        t["cost_usd"] += cost
        t["latency_s"] += latency
        if resp and not hit:
            for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
                t[k] += int(resp["usage"].get(k) or 0)

    def _write(self, record):
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()


class RandomPolicy:
    """The L3 control: on the same draws, a random allowed condition the
    lab has not run instead of the LLM's pick."""

    def __init__(self, seed, alpha=0.3, warmup=5):
        self.seed, self.alpha, self.warmup = int(seed), float(alpha), warmup
        self.totals = {"calls": 0}

    def wants(self, rnd, agent):
        return coin(self.seed, agent, rnd, self.alpha, self.warmup)

    def decide(self, contexts):
        out = []
        for ctx in contexts:
            tried = {tuple(int(v) for v in x) for x in ctx["observed"]}
            free = [x for x in itertools.product(
                *[[int(i) for i in a] for a in ctx["allowed"]])
                if x not in tried]
            rng = np.random.default_rng(
                [self.seed, int(ctx["agent"]), int(ctx["round"]), 5])
            self.totals["calls"] += 1
            x = list(free[rng.integers(len(free))]) if free else None
            out.append({"x": x, "source": "random", "insight": None,
                        "call_id": None})
        return out

    def record_outcome(self, call_id, y, q):
        pass

    def close(self):
        pass


def build_policy(args, run_dir):
    """args: llm (l3 or random), alpha, llm_model, llm_effort, llm_share,
    llm_solo, llm_base_url, llm_max_tokens, max_usd, seed, setting,
    agents[, label]; APPFL adds llm_api_key and log_name.

    DRY_RUN=1 (or llm_model='mock') swaps in the offline MockClient; a base
    URL, a local OpenAI-compatible server (vLLM) for the OpenAI API.
    """
    get = (args.get if isinstance(args, dict)
           else lambda k, d=None: getattr(args, k, d))
    alpha = 0.3 if get("alpha") is None else get("alpha")
    # APPFL: warmup is configurable here (the reference fixes it at 5), so 0 must stay 0
    # and the control's coin must pair with L3's.
    warmup = 5 if get("warmup_rounds") is None else int(get("warmup_rounds"))
    if get("llm") == "random":
        return RandomPolicy(get("seed"), alpha, warmup)
    model = get("llm_model") or "gpt-6.1-sol"
    mock = os.environ.get("DRY_RUN") == "1" or model == "mock"
    url = get("llm_base_url")
    client = (MockClient() if mock
              else LocalClient(url, api_key=get("llm_api_key") or "local")
              if url else OpenAIClient())  # APPFL: the key, for a gateway
    label = get("label")
    return L3Policy(
        client, run_dir, seed=get("seed"), setting=get("setting"),
        alpha=alpha, model="gpt-6.1-sol" if model == "mock" else model,
        effort=get("llm_effort") or "high", max_usd=get("max_usd"),
        n_agents=get("agents") or get("n_agents") or 4,
        warmup=warmup,  # APPFL
        run_id=f"{label}-{get('setting')}-s{get('seed')}" if label else None,
        share=get("llm_share") or "hypothesis", solo=bool(get("llm_solo")),
        max_tokens=get("llm_max_tokens") or 25000,
        log_name=get("log_name") or "llm_calls.jsonl")  # APPFL
