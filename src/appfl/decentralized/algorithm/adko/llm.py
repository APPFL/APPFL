"""OpenAI-compatible language model for ADKO's two LM call sites.

ADKO uses a language model at exactly two points in Algorithm 1, and both are optional:

* **step 3, PROPOSE** -- given accumulated peer tokens, suggest candidate design points. This
  is where cross-slice transfer happens: the model reads that a motif worked in one agent's
  region and proposes its analogue in this agent's. In the reference's chemistry study it
  cuts the scored candidate set from ~3,696 to 10 per round at comparable hit rate.
* **step 10, ENCODE** -- write the one-sentence ``z`` insight carried by an outgoing token.

The paper ablates both: its NAS study runs with no LM at all, isolating token-based
collaboration from semantic reasoning. Keep that switch working. ``LLMConfig.enabled=False``
means :func:`build_language_model` returns ``None`` and :class:`ADKOAgent` takes the LM-free
path with no other change.

Points at any OpenAI-compatible endpoint via ``base_url`` -- vLLM, Argo, llama.cpp, a
gateway -- which is not a convenience but a requirement; see the privacy note below.

.. warning::
   **The ENCODE prompt contains the raw observation ``y``.** It has to: the model is being
   asked *why* a result came out the way it did. ADKO Constraint 3.1 governs what crosses
   between *agents*, and the LM is the agent's own tool, so this is not a violation of the
   algorithm -- but it does mean the endpoint sees raw data. For any federation holding
   proprietary or restricted data, point ``base_url`` at a locally hosted model. Sending
   Bayer's sorghum yields or an unpublished alloy measurement to a third-party API is a data
   egress event regardless of what the algorithm guarantees. :attr:`LLMConfig.allow_remote`
   exists to make that choice explicit rather than accidental.

   The PROPOSE prompt is safer by construction: it carries peer tokens (``s``, ``c``, ``eta``,
   the noised design point, ``z``) and this agent's own *embedded* history, never raw
   observations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken, Signal
from appfl.decentralized.algorithm.adko.components import DesignSpace, LanguageModel

LOCAL_HOST_HINTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", ".local", ".gov")


@dataclass
class LLMConfig:
    """Everything needed to reach a model, and the switches to not reach one at all.

    :param enabled: master switch. ``False`` means no LM anywhere -- the ablation arm.
    :param base_url: OpenAI-compatible endpoint. ``None`` uses the OpenAI default. Set this
        to a local vLLM/Argo/gateway URL for restricted data.
    :param api_key: bearer token. Falls back to ``$APPFL_DECENTRALIZED_LLM_API_KEY`` then
        ``$OPENAI_API_KEY``. Never logged, never written to the cache.
    :param allow_remote: guard rail. When ``False`` (the default) a non-local ``base_url``
        raises rather than silently shipping raw observations off-site.
    :param propose / emit_insight: the two call sites, switchable independently so the
        contribution of each can be ablated.
    :param cache_path: SQLite prompt cache. Strongly recommended -- multi-seed runs repeat
        prompts heavily, and the reference's own caches are excluded from its repo for size,
        which is what makes reproducing its LLM arms expensive.
    """

    enabled: bool = False
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: Optional[float] = 0.7
    max_output_tokens: Optional[int] = 512
    #: Which parameter name carries the output cap. Newer reasoning models on some gateways
    #: reject ``max_tokens`` and want ``max_completion_tokens``; set ``None`` to send neither
    #: and let the server decide, which is what an unfamiliar proxy usually wants.
    token_limit_param: Optional[str] = "max_tokens"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_base: float = 2.0
    allow_remote: bool = False
    propose: bool = True
    emit_insight: bool = True
    cache_path: Optional[str] = None
    verbose: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        """Build from environment variables, then apply explicit overrides.

        ``APPFL_DECENTRALIZED_LLM_{ENABLED,BASE_URL,API_KEY,MODEL,ALLOW_REMOTE,CACHE}``, with
        ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` as fallbacks. Environment rather than
        config files for the key specifically, so it never lands in a checked-in YAML.
        """

        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")

        config = cls(
            enabled=flag("APPFL_DECENTRALIZED_LLM_ENABLED", False),
            base_url=os.environ.get("APPFL_DECENTRALIZED_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("APPFL_DECENTRALIZED_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("APPFL_DECENTRALIZED_LLM_MODEL", "gpt-4o-mini"),
            allow_remote=flag("APPFL_DECENTRALIZED_LLM_ALLOW_REMOTE", False),
            cache_path=os.environ.get("APPFL_DECENTRALIZED_LLM_CACHE"),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    def is_local_endpoint(self) -> bool:
        """Whether ``base_url`` points at this machine or an internal host.

        .. warning::
           This checks the *address*, not where the data ends up. A proxy on
           ``http://localhost:PORT/v1`` that forwards to a hosted model is indistinguishable
           from a model actually running on the box, and this returns ``True`` for both. If
           your endpoint is a tunnel to something external, the raw observations in the
           ENCODE prompt still leave the building -- decide that deliberately rather than
           letting the address decide for you.
        """
        if not self.base_url:
            return False  # the OpenAI default is by definition remote
        return any(hint in self.base_url.lower() for hint in LOCAL_HOST_HINTS)

    def validate(self) -> None:
        """Fail fast on a configuration that would leak, rather than mid-run."""
        if not self.enabled:
            return
        if not self.api_key:
            raise ValueError(
                "LLM is enabled but no API key was found. Set "
                "APPFL_DECENTRALIZED_LLM_API_KEY (or OPENAI_API_KEY), or pass --llm-api-key. "
                "Gateways that authenticate by identity rather than a secret take the "
                "identity here (a username works); endpoints that ignore auth entirely still "
                "need a placeholder such as 'none', because the OpenAI SDK requires one."
            )
        if self.emit_insight and not self.is_local_endpoint() and not self.allow_remote:
            endpoint = self.base_url or "the OpenAI API"
            raise ValueError(
                f"Insight generation sends raw observations to {endpoint}, which is outside "
                "this deployment. Either point --llm-base-url at a locally hosted model, "
                "pass --llm-allow-remote to accept the egress deliberately, or use "
                "--llm-no-insight to keep only candidate proposal, which never sees raw "
                "observations."
            )


class _PromptCache:
    """SQLite prompt cache keyed on (model, prompt). Survives across runs and seeds.

    Worth having: a 40-seed sweep repeats prompts heavily, and re-querying a paid API for
    results you already have is the single largest avoidable cost in reproducing an LLM arm.

    Safe to share across MPI ranks -- WAL plus a busy timeout handles the concurrent writers,
    and a shared cache is the point, since neighbouring agents ask overlapping questions. Give
    ranks separate paths only if the shared file lands on a filesystem where SQLite locking is
    unreliable, which on most parallel filesystems it is.
    """

    def __init__(self, path: str, busy_timeout: float = 30.0):
        import sqlite3

        self._sqlite3 = sqlite3
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=busy_timeout)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS prompts ("
            "  key TEXT PRIMARY KEY, kind TEXT, model TEXT, response TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, prompt: str) -> str:
        import hashlib

        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT response FROM prompts WHERE key = ?", (self.key(model, prompt),)
        ).fetchone()
        return row[0] if row else None

    def put(self, model: str, prompt: str, response: str, kind: str) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO prompts (key, kind, model, response) VALUES (?,?,?,?)",
                (self.key(model, prompt), kind, model, response),
            )
            self._conn.commit()
        except self._sqlite3.OperationalError:
            # A contended cache write is not worth failing a run over; the next caller just
            # pays for the call again.
            pass


def _parse_json_object(text: str) -> Optional[Any]:
    """Pull a JSON object or array out of a model response.

    Models wrap JSON in prose and fenced code blocks no matter how firmly the prompt asks
    them not to, so try the strict parse first and fall back to extracting the outermost
    braces or brackets. Returns ``None`` rather than raising -- a malformed response is an
    expected event, not an error.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class OpenAILanguageModel(LanguageModel):
    """:class:`LanguageModel` over any OpenAI-compatible chat-completions endpoint.

    Every failure path degrades rather than raises: a timeout, a rate limit, or an
    unparseable response yields an empty proposal list or a ``None`` insight, and the agent
    continues on the LM-free path. A 40-round federated run must not die because one endpoint
    hiccuped in round 12.
    """

    def __init__(self, config: LLMConfig, agent_id: str = ""):
        config.validate()
        self.config = config
        self.agent_id = agent_id
        self.name = config.model
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self._cache = _PromptCache(config.cache_path) if config.cache_path else None
        self._client = None  # built lazily so importing this module needs no openai

    # -- transport ---------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "LLM support needs the openai package: pip install openai"
                ) from exc
            kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout_seconds,
                "max_retries": 0,  # retries handled here, so backoff is observable
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _complete(self, prompt: str, kind: str) -> Optional[str]:
        """One chat completion, cached and retried. ``None`` on give-up."""
        import time

        if self._cache is not None:
            cached = self._cache.get(self.config.model, prompt)
            if cached is not None:
                self.cache_hits += 1
                return cached

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                # Only send parameters that were actually configured. Company gateways and
                # reasoning models are inconsistent here: some reject a non-default
                # ``temperature`` outright, some want ``max_completion_tokens`` instead of
                # ``max_tokens``, and some reject an unrecognised key rather than ignoring
                # it. Omitting what you have not set is the portable choice.
                call_kwargs = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.config.temperature is not None:
                    call_kwargs["temperature"] = self.config.temperature
                if self.config.max_output_tokens is not None and self.config.token_limit_param:
                    call_kwargs[self.config.token_limit_param] = self.config.max_output_tokens
                call_kwargs.update(self.config.extra_body)
                response = self._get_client().chat.completions.create(**call_kwargs)
                text = (response.choices[0].message.content or "").strip()
                self.calls += 1
                if self._cache is not None:
                    self._cache.put(self.config.model, prompt, text, kind)
                return text
            except Exception as exc:  # noqa: BLE001 - any endpoint failure is recoverable
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.backoff_base**attempt)

        self.failures += 1
        if self.config.verbose:
            print(f"[llm:{self.agent_id}] {kind} failed after "
                  f"{self.config.max_retries} attempts: {last_error}")
        return None

    # -- LanguageModel -----------------------------------------------------------------

    def propose(
        self,
        token_memory: Sequence[KnowledgeToken],
        space: DesignSpace,
        n: int,
        history: Optional[Sequence[Any]] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Algorithm 1 step 3. Peer tokens in, candidate design points out.

        Carries no raw observations -- only what peers already published plus the space
        description. Anything the model returns that ``space.parse`` rejects is dropped, so a
        model that ignores the slice restriction cannot smuggle out-of-range candidates into
        the scored set.
        """
        if not self.config.propose or n <= 0:
            return []
        prompt = self._propose_prompt(token_memory, space, n, history, progress)
        text = self._complete(prompt, kind="propose")
        if text is None:
            return []
        parsed = _parse_json_object(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("candidates", [])
        if not isinstance(parsed, list):
            return []

        points: List[Any] = []
        for item in parsed[:n]:
            try:
                point = space.parse(item)
            except NotImplementedError:
                raise
            except Exception:  # noqa: BLE001 - a bad candidate is data, not a crash
                point = None
            if point is not None:
                points.append(point)
        return points

    def encode_insight(
        self, embedding: Sequence[float], observation: float, threshold: float
    ) -> Optional[str]:
        """Algorithm 1 step 10. One sentence explaining the outcome, for peers to reuse.

        Three regimes, following the reference: confident success, confident failure, and a
        near-threshold band where the right answer is to say the result is uninformative
        rather than to confabulate a mechanism. That last case matters -- a low-``c`` token
        already carries almost no numeric signal, and an invented explanation attached to it
        is worse than nothing, because unlike a bad number a bad sentence propagates.
        """
        if not self.config.emit_insight:
            return None
        prompt = self._insight_prompt(embedding, observation, threshold)
        text = self._complete(prompt, kind="insight")
        if text is None:
            return None
        parsed = _parse_json_object(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("insight"), str):
            return parsed["insight"].strip() or None
        # A model that answered with a bare sentence is still useful; keep it if it is short
        # enough to be an insight rather than an essay.
        line = text.strip().strip('"')
        return line if 0 < len(line) <= 400 else None

    # -- prompts -----------------------------------------------------------------------

    @staticmethod
    def _format_progress(progress: Optional[Dict[str, Any]]) -> str:
        if not progress:
            return "  (no observations yet)"
        best = progress.get("best_y")
        return (
            f"  observations={progress.get('n_obs', 0)}, "
            f"best={'n/a' if best is None else f'{best:.3f}'}, "
            f"rounds_since_improvement={progress.get('rounds_since_improve', 0)}, "
            f"recent_improvement={float(progress.get('recent_improvement', 0.0)):.3f}"
        )

    @staticmethod
    def _format_history(history: Optional[Sequence[Any]], limit: int = 8) -> str:
        """This laboratory's own results. Local -- it goes to the agent's own model, never
        to a peer, and it is what lets the model exploit rather than only react."""
        if not history:
            return "  (nothing measured yet)"
        recent = list(history)[-limit:]
        return "\n".join(f"  {point!r} -> {value:.3f}" for point, value in recent)

    def _propose_prompt(
        self,
        token_memory: Sequence[KnowledgeToken],
        space: DesignSpace,
        n: int,
        history: Optional[Sequence[Any]] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> str:
        return (
            "You are helping one laboratory in a federation choose which experiments to run "
            "next. Each laboratory searches its own region and shares only compact summaries "
            "of its results.\n\n"
            f"## What this laboratory may choose\n{space.describe()}\n\n"
            f"## Search progress\n{self._format_progress(progress)}\n\n"
            f"## What this laboratory has already measured\n"
            f"{self._format_history(history)}\n\n"
            f"## What peer laboratories have reported\n"
            f"{self._format_peer_tokens(token_memory)}\n\n"
            "## Task\n"
            f"Propose {n} candidate experiments for THIS laboratory. Stay strictly inside the "
            "region described above, and do not repeat anything already measured.\n\n"
            "Cover a mix of intents rather than all of one kind: some that exploit near this "
            "laboratory's own best result, some that transfer a mechanism a peer reported, "
            "and some that explore regions nobody has touched. If progress has stalled, "
            "weight exploration higher. The value in a peer's report is the reason, not the "
            "coordinates -- do not simply copy a peer's design point.\n\n"
            "## Output\n"
            'Strict JSON, no prose outside it: {"candidates": [<candidate>, ...]}'
        )

    def _insight_prompt(
        self, embedding: Sequence[float], observation: float, threshold: float
    ) -> str:
        deviation = observation - threshold
        near_threshold = abs(deviation) < 0.3 * max(abs(threshold), 1e-8)
        if near_threshold:
            task = (
                "Write ONE sentence (<= 30 words) flagging this result as near-threshold and "
                "uninformative for peers. Do NOT speculate about mechanism. Say what was "
                "tried and that the outcome is ambiguous."
            )
        else:
            task = (
                "Write ONE sentence (<= 30 words) giving the likely reason this result "
                "succeeded or failed, phrased so another laboratory searching a different "
                "region could act on it. Describe the mechanism, not the coordinates. Do not "
                "invent numbers you were not shown."
            )
        return (
            "You are summarising one experimental result for peer laboratories that cannot "
            "see your data.\n\n"
            "## The result\n"
            f"  design point (embedded) : {[round(float(v), 4) for v in embedding]}\n"
            f"  measured value          : {observation:.4f}\n"
            f"  threshold               : {threshold:.4f}\n"
            f"  outcome                 : "
            f"{'SUCCESS' if deviation >= 0 else 'FAILURE'}"
            f"{' (near threshold, weak evidence)' if near_threshold else ''}\n\n"
            f"## Task\n{task}\n\n"
            "## Output\n"
            'Strict JSON, no prose outside it: {"insight": "<your single sentence>"}'
        )

    @staticmethod
    def _format_peer_tokens(token_memory: Sequence[KnowledgeToken], limit: int = 12) -> str:
        """Render peer tokens for the prompt. Only published fields, never local state.

        Ordered by fidelity so the strongest evidence survives the context budget, and
        balanced across successes and failures -- a prompt showing only successes produces a
        model that never learns what to avoid.
        """
        if not token_memory:
            return "  (no peer reports yet)"
        ranked = sorted(token_memory, key=lambda t: t.fidelity(), reverse=True)
        successes = [t for t in ranked if t.signal is Signal.SUCCESS][: limit // 2]
        failures = [t for t in ranked if t.signal is Signal.FAIL][: limit // 2]
        lines = []
        for token in successes + failures:
            line = (
                f"  [from {token.provenance.agent_id or 'peer'}, round "
                f"{token.provenance.round}, {token.signal.value.upper()}, "
                f"strength={token.advantage:.2f}, fidelity={token.fidelity():.2f}] "
                f"near {[round(float(v), 4) for v in token.embedding]}"
            )
            if token.insight:
                line += f'\n      "{token.insight}"'
            lines.append(line)
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        """Call accounting, for reporting what an LLM arm actually cost."""
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
        }


def build_language_model(
    config: LLMConfig, agent_id: str = ""
) -> Optional[OpenAILanguageModel]:
    """Return a model, or ``None`` when disabled.

    The single switch the rest of the code needs: ``ADKOAgent(language_model=None)`` is the
    LM-free ablation arm, so turning the LLM off is a configuration change and not a code
    path.
    """
    if not config.enabled:
        return None
    return OpenAILanguageModel(config, agent_id=agent_id)
