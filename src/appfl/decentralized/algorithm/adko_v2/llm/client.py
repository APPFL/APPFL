"""Responses API client, a local (vLLM) client, a deterministic mock, the
call cache, prices."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import itertools
import json
import random
import sqlite3

import numpy as np

from .schema import FORMAT, KEYS

# $ per million tokens (input, output); output includes reasoning.
PRICES = {"gpt-6.1-sol": (2.0, 10.0),
          "gpt-5.4-mini-2026-03-17": (0.75, 4.50)}


def cost_usd(model, usage):
    p_in, p_out = PRICES.get(model, PRICES["gpt-6.1-sol"])
    return (usage.get("input_tokens", 0) * p_in
            + usage.get("output_tokens", 0) * p_out) / 1e6


def make_request(model, effort, system, user, max_output_tokens=25000):
    return {"model": model, "instructions": system, "input": user,
            "reasoning": {"effort": effort, "summary": "auto"},
            "text": {"format": FORMAT}, "store": False,
            "max_output_tokens": max_output_tokens}


def request_key(request, client_name):
    blob = json.dumps({"client": client_name, **request}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    """Prompt -> response store in the run dir (reruns cost nothing)."""

    def __init__(self, path):
        # APPFL: a busy timeout, since every lab -- possibly in its own MPI rank --
        # shares this file.
        self.db = sqlite3.connect(str(path), timeout=30.0)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls "
                        "(key TEXT PRIMARY KEY, response TEXT)")

    def get(self, key):
        row = self.db.execute("SELECT response FROM calls WHERE key = ?",
                              (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, response):
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)",
                        (key, json.dumps(response)))
        self.db.commit()

    def close(self):
        self.db.close()


def unpack(resp):
    text, summary = [], []
    for item in resp.output or []:
        if item.type == "message":
            text += [p.text for p in item.content or []
                     if p.type == "output_text"]
        elif item.type == "reasoning":
            summary += [s.text for s in item.summary or []]
    u, usage = resp.usage, {}
    if u is not None:
        usage = {"input_tokens": u.input_tokens,
                 "output_tokens": u.output_tokens,
                 "reasoning_tokens": getattr(u.output_tokens_details,
                                             "reasoning_tokens", 0),
                 "cached_tokens": getattr(u.input_tokens_details,
                                          "cached_tokens", 0)}
    return {"text": "".join(text), "summary": "\n\n".join(summary),
            "usage": usage, "response_id": resp.id, "status": resp.status}


class OpenAIClient:
    name = "openai"

    def __init__(self, concurrency=4, timeout=900.0, retries=4,
                 backoff=4.0, backoff_max=120.0):
        from openai import AsyncOpenAI
        self.api = AsyncOpenAI(max_retries=0, timeout=timeout)
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout, self.retries = timeout, retries
        self.backoff, self.backoff_max = backoff, backoff_max

    async def complete(self, request, ctx=None):
        import openai
        transient = (openai.RateLimitError, openai.APITimeoutError,
                     openai.APIConnectionError, openai.InternalServerError,
                     asyncio.TimeoutError)
        async with self.sem:
            for attempt in range(self.retries + 1):
                try:
                    resp = await asyncio.wait_for(
                        self.api.responses.create(**request), self.timeout)
                    return unpack(resp)
                except openai.BadRequestError as exc:
                    # Fall back to free-form JSON if the schema is rejected.
                    if "text" not in request or "format" not in str(exc):
                        raise
                    request = {k: v for k, v in request.items()
                               if k != "text"}
                except transient:
                    if attempt == self.retries:
                        raise
                    delay = min(self.backoff * 2 ** attempt,
                                self.backoff_max)
                    await asyncio.sleep(delay * (1 + 0.25 * random.random()))
            raise RuntimeError("retries exhausted")

    async def close(self):
        await self.api.close()


def chat_request(request):
    """A make_request() request as an OpenAI-style chat completion, the API
    vLLM serves; the seed follows the prompt, so reruns repeat it."""
    body = {"model": request["model"],
            "messages": [
                {"role": "system", "content": request["instructions"]},
                {"role": "user", "content": request["input"]}],
            "max_tokens": request["max_output_tokens"],
            "seed": int(request_key(request, "local")[:8], 16)}
    if "text" in request:
        fmt = request["text"]["format"]
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": fmt["name"], "schema": fmt["schema"], "strict": True}}
    if "gpt-oss" in request["model"]:
        body["reasoning_effort"] = request["reasoning"]["effort"]
    # APPFL: adko_h sends per-server settings (temperature, thinking) this way. Absent from
    # v2's own requests, so their cache keys and bodies are unchanged.
    body.update(request.get("extra") or {})
    return body


def unpack_chat(resp):
    """Text, the full reasoning trace and usage of a chat completion."""
    choice = resp.choices[0]
    msg, extra = choice.message, choice.message.model_extra or {}
    trace = extra.get("reasoning_content") or extra.get("reasoning") or ""
    u = resp.usage
    usage = {} if u is None else {
        "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
        "reasoning_tokens": 0, "cached_tokens": 0}
    return {"text": msg.content or "", "summary": None, "reasoning": trace,
            "usage": usage, "response_id": resp.id,
            "status": choice.finish_reason}


def sdk_body(create, body):
    """APPFL: ``body`` as ``create(**body)`` takes it. Fields the SDK has no
    argument for (vLLM's ``chat_template_kwargs``) raise TypeError before any
    request is sent, so they move to ``extra_body``, which the SDK merges into
    the JSON: the server receives the same fields."""
    known = inspect.signature(create).parameters
    extra = {k: body.pop(k) for k in [k for k in body if k not in known]}
    if extra:
        body["extra_body"] = {**body.get("extra_body", {}), **extra}
    return body


class LocalClient:
    """An open model on our own GPU behind vLLM's OpenAI-compatible server:
    free, and it returns the full reasoning trace."""

    name, free = "local", True

    def __init__(self, base_url, concurrency=16, timeout=3600.0,
                 api_key="local"):  # APPFL: a gateway needs a real key
        from openai import AsyncOpenAI
        self.api = AsyncOpenAI(base_url=base_url, api_key=api_key,
                               max_retries=3, timeout=timeout)
        self.sem = asyncio.Semaphore(concurrency)

    async def complete(self, request, ctx=None):
        import openai
        body = sdk_body(self.api.chat.completions.create, chat_request(request))
        async with self.sem:
            try:
                resp = await self.api.chat.completions.create(**body)
            except openai.BadRequestError:
                if "response_format" not in body:
                    raise  # else retry as free-form JSON, parsed leniently
                body.pop("response_format")
                resp = await self.api.chat.completions.create(**body)
        return unpack_chat(resp)

    async def close(self):
        await self.api.close()


class MockClient:
    """Valid, deterministic answers with typical usage; no network."""

    name = "mock"
    usage = {"input_tokens": 6000, "output_tokens": 6500,
             "reasoning_tokens": 6000, "cached_tokens": 0}

    def __init__(self):
        self.calls = 0

    async def complete(self, request, ctx):
        self.calls += 1
        tried = {tuple(int(v) for v in x) for x in ctx["observed"]}
        free = [x for x in itertools.product(
            *[[int(i) for i in a] for a in ctx["allowed"]]) if x not in tried]
        rng = np.random.default_rng([int(ctx["agent"]), int(ctx["round"])])
        n = min(3, len(free))
        picks = [free[i] for i in rng.choice(len(free), n, replace=False)]
        picks = [dict(zip(KEYS, x)) for x in picks or [(0,) * 5]]
        body = {"neighbor_analysis": "mock analysis",
                "evidence_ids": [t["id"] for t in ctx["memory"][:2]],
                "hypothesis": f"mock hypothesis r{ctx['round']}",
                "strategy": "mock strategy", "intent": "test",
                "expected_yield": [40.0, 60.0],
                "proposal": picks[0], "fallbacks": picks[1:3]}
        return {"text": json.dumps(body), "summary": "mock reasoning",
                "usage": dict(self.usage),
                "response_id": f"mock-{self.calls}", "status": "completed"}

    async def close(self):
        pass
