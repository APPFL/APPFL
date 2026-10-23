"""The AmSC Model Access Gateway (MAG) as a campaign planner: frontier models, paid per call.

MAG is an OpenAI-compatible gateway (LiteLLM underneath) to commercial models and DOE-hosted
open ones. It is the other way to run ADKO-H's planner -- no GPU, no served model, any model it
lists -- and it differs from a vLLM server in four ways this module exists to handle:

* **it costs money, per call.** The gateway reports each call's charge in a response header;
  every charge goes to a ledger file, and no call starts once the ledger reaches the cap. A run
  that hits the cap stops with :class:`BudgetSpent` instead of quietly continuing with random
  designs, which would turn the LLM arm into the control arm without saying so.
* **it is rate-limited per user, not per job.** At the time of writing: 20 requests in flight and
  120 a minute, across everything the user runs. Requests therefore go through slot files that
  every process on the node shares; each slot starts at most one request per ``MAG_GAP``
  seconds, so ``MAG_SLOTS=18, MAG_GAP=10`` stays under both limits with a margin. The slots are
  local files, so **runs on several nodes do not see each other's slots** -- run every MAG arm
  from one node, or divide ``MAG_SLOTS`` between the nodes.
* **providers reject what vLLM accepts.** ``seed`` and vLLM's ``chat_template_kwargs`` are
  dropped; reasoning effort is chosen by the model id (``claude-opus-5-5-high@amazon``), not by a
  request field; and a schema the provider refuses is retried once as free-form JSON.
* **the token is a credential.** It comes from ``AMSC_TOKEN`` or the file named by
  ``AMSC_TOKEN_FILE`` and is never logged, never written to a run record, never printed.

Command line, for checking a node before a run (each costs at most a few cents)::

    python -m appfl.decentralized.algorithm.adko_h.llm.mag spend
    python -m appfl.decentralized.algorithm.adko_h.llm.mag ping --model claude-haiku-5-5@amazon
    python -m appfl.decentralized.algorithm.adko_h.llm.mag models --grep opus
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request, unpack_chat

URL = "https://mag.genesis.americansciencecloud.org"
#: The header LiteLLM puts each call's charge in, in dollars.
COST = "x-litellm-response-cost"
#: Request fields a served vLLM model takes and a commercial provider may reject.
VLLM_ONLY = ("seed", "chat_template_kwargs", "reasoning_effort")


def base_url() -> str:
    return os.environ.get("AMSC_MAG_URL", URL).rstrip("/")


def token() -> str:
    """The AmSC token, from the environment. Never logged; never in an exception message."""
    if os.environ.get("AMSC_TOKEN"):
        return os.environ["AMSC_TOKEN"].strip()
    path = os.environ.get("AMSC_TOKEN_FILE")
    if not path:
        raise RuntimeError("set AMSC_TOKEN or AMSC_TOKEN_FILE to use MAG")
    return Path(path).expanduser().read_text().strip()


class Slots:
    """At most ``n`` requests in flight among this node's processes; a slot starts a request
    at most once per ``gap`` seconds. Local files, so they coordinate one node, not a cluster."""

    def __init__(self, n: int = 18, gap: float = 10.0, root: Optional[str] = None):
        root_path = Path(root or f"/tmp/mag_slots-{os.getuid()}")
        root_path.mkdir(parents=True, exist_ok=True)
        self.paths = [root_path / f"slot{i}" for i in range(int(n))]
        self.gap = float(gap)

    async def acquire(self) -> int:
        while True:
            for path in random.sample(self.paths, len(self.paths)):
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                last = float(os.pread(fd, 32, 0).strip() or 0)
                await asyncio.sleep(max(0.0, last + self.gap - time.time()))
                os.pwrite(fd, f"{time.time():<32.3f}".encode(), 0)
                return fd
            await asyncio.sleep(0.5 + random.random())

    @staticmethod
    def release(fd: int) -> None:
        os.close(fd)  # closing the descriptor drops the lock


class Ledger:
    """Dollars spent by every run that shares ``path``, one line per call; a hard cap."""

    def __init__(self, path: str, cap: Optional[float] = None):
        self.path, self.cap = Path(path), cap
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def add(self, usd: float) -> None:
        with open(self.path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{float(usd):.6f}\n")

    def total(self) -> float:
        return sum(float(v) for v in self.path.read_text().split())

    def over(self) -> bool:
        return self.cap is not None and self.total() >= self.cap


class BudgetSpent(RuntimeError):
    """The ledger reached its cap. The planner re-raises this rather than falling back."""


class MagClient:
    """A campaign planner's client for MAG: same ``complete`` contract as ``LocalClient``.

    Configured from the environment so a job script, not the code, decides the money:
    ``MAG_LEDGER`` (ledger file), ``MAG_BUDGET`` (dollar cap on that ledger), ``MAG_SLOTS``,
    ``MAG_GAP``, ``MAG_SLOT_DIR``, ``AMSC_MAG_URL``.
    """

    name, free = "mag", False

    def __init__(self, slots: Optional[Slots] = None, ledger: Optional[Ledger] = None,
                 timeout: float = 600.0, retries: int = 6, api: Any = None):
        env = os.environ.get
        if api is None:
            from openai import AsyncOpenAI

            api = AsyncOpenAI(base_url=base_url() + "/v1", api_key=token(), max_retries=0,
                              timeout=timeout)
        self.api = api
        self.slots = slots or Slots(int(env("MAG_SLOTS", 18)), float(env("MAG_GAP", 10)),
                                    env("MAG_SLOT_DIR"))
        if ledger is None and env("MAG_LEDGER"):
            cap = env("MAG_BUDGET")
            ledger = Ledger(env("MAG_LEDGER"), float(cap) if cap else None)
        self.ledger, self.retries = ledger, int(retries)

    async def complete(self, request: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None):
        import openai

        body = chat_request(request)
        for key in VLLM_ONLY:
            body.pop(key, None)
        transient = (openai.RateLimitError, openai.APITimeoutError,
                     openai.APIConnectionError, openai.InternalServerError)
        for attempt in range(self.retries + 1):
            if self.ledger is not None and self.ledger.over():
                raise BudgetSpent(f"MAG ledger reached its cap of ${self.ledger.cap:.2f}")
            fd = await self.slots.acquire()
            try:
                raw = await self.api.chat.completions.with_raw_response.create(**body)
            except openai.BadRequestError as exc:
                if "response_format" in body:
                    body.pop("response_format")  # retry once as free-form JSON
                    continue
                if "temperature" in body and "temperature" in str(exc).lower():
                    body.pop("temperature")  # e.g. a thinking model that fixes it
                    continue
                raise
            except transient:
                if attempt == self.retries:
                    raise
                await asyncio.sleep(min(5 * 2**attempt, 120) * (1 + 0.25 * random.random()))
                continue
            finally:
                self.slots.release(fd)
            out = unpack_chat(raw.parse())
            out["usage"]["cost_usd"] = float(raw.headers.get(COST) or 0.0)
            if self.ledger is not None:
                self.ledger.add(out["usage"]["cost_usd"])
            return out
        raise RuntimeError("MAG: retries exhausted")

    async def close(self) -> None:
        closer = getattr(self.api, "close", None)
        if closer is not None:
            await closer()


def spend() -> Dict[str, Any]:
    """What the gateway says this user has spent, and the caps it reports.

    ``/user/info`` is LiteLLM's; MAG reports a personal 30-day budget and the team (project)
    budget. Which one actually stops calls is the gateway's business -- treat both as
    information, and the local ledger cap as the rule.
    """
    req = urllib.request.Request(
        base_url() + "/user/info", headers={"Authorization": f"Bearer {token()}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read().decode())
    u = d.get("user_info") or {}
    return {
        "spend_usd": u.get("spend"),
        "personal_max_budget_usd": u.get("max_budget"),
        "budget_duration": u.get("budget_duration"),
        "budget_reset_at": u.get("budget_reset_at"),
        "teams": [
            {k: t.get(k) for k in ("team_alias", "spend", "max_budget", "budget_duration")}
            for t in d.get("teams") or []
        ],
    }


def _ping(model: str) -> Dict[str, Any]:
    async def go():
        client = MagClient(retries=1)
        t0 = time.time()
        req = {
            "model": model,
            "instructions": "Reply with exactly: ok",
            "input": "ping",
            "reasoning": {"effort": "low"},
            "max_output_tokens": 64,
        }
        try:
            out = await client.complete(req)
        finally:
            await client.close()
        return {
            "model": model,
            "reply": (out.get("text") or "").strip()[:40],
            "seconds": round(time.time() - t0, 2),
            "cost_usd": out["usage"].get("cost_usd"),
        }

    return asyncio.run(go())


def _models(grep: Optional[str]) -> list:
    req = urllib.request.Request(
        base_url() + "/v1/models", headers={"Authorization": f"Bearer {token()}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        ids = sorted(m["id"] for m in json.loads(resp.read().decode()).get("data", []))
    return [i for i in ids if not grep or grep in i]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="adko_h.llm.mag", description="Check MAG from this node.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spend", help="what the gateway says you have spent")
    s = sub.add_parser("ping", help="one tiny call: reachability, latency, cost")
    s.add_argument("--model", default="claude-haiku-5-5@amazon")
    s = sub.add_parser("models", help="model ids the gateway lists")
    s.add_argument("--grep", default=None)
    args = p.parse_args(argv)
    if args.cmd == "spend":
        print(json.dumps(spend(), indent=2))
    elif args.cmd == "ping":
        print(json.dumps(_ping(args.model)))
    else:
        print("\n".join(_models(args.grep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
