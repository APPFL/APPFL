"""L3 output schema, parsing, and choice of the executed condition."""

from __future__ import annotations

import re

from .llm_suzuki import _parse_json_response

KEYS = ("electrophile_idx", "nucleophile_idx", "base_idx", "ligand_idx",
        "solvent_idx")
INTENTS = ("transfer", "refine", "test", "explore")

_TUPLE = {"type": "object",
          "properties": {k: {"type": "integer"} for k in KEYS},
          "required": list(KEYS), "additionalProperties": False}
_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "neighbor_analysis": _STR,
        "evidence_ids": {"type": "array", "items": _STR},
        "hypothesis": _STR,
        "strategy": _STR,
        "intent": {"type": "string", "enum": list(INTENTS)},
        "expected_yield": {"type": "array", "items": {"type": "number"}},
        "proposal": _TUPLE,
        "fallbacks": {"type": "array", "items": _TUPLE},
    },
    "additionalProperties": False,
}
SCHEMA["required"] = list(SCHEMA["properties"])
FORMAT = {"type": "json_schema", "name": "l3_decision", "schema": SCHEMA,
          "strict": True}


def parse(text):
    parsed = _parse_json_response(text or "")
    return parsed if isinstance(parsed, dict) else None


def as_tuple(obj):
    try:
        return [int(obj[k]) for k in KEYS]
    except (KeyError, TypeError, ValueError):
        return None


def snap(x, allowed):
    """Force single-option (fixed) dims; None if another dim is invalid."""
    out = list(x)
    for d, idxs in enumerate(allowed):
        idxs = [int(i) for i in idxs]
        if out[d] not in idxs:
            if len(idxs) != 1:
                return None, False
            out[d] = idxs[0]
    return out, out != list(x)


def choose(parsed, ctx):
    """(x, source, reason): proposal, then fallbacks; x None if none fit."""
    tried = {tuple(int(v) for v in x) for x in ctx["observed"]}
    fallbacks = parsed.get("fallbacks") or []
    options = [("proposal", parsed.get("proposal"))]
    options += [(f"fallback{i + 1}", fb)
                for i, fb in enumerate(fallbacks[:2])]
    notes = []
    for source, obj in options:
        x = as_tuple(obj) if isinstance(obj, dict) else None
        if x is None:
            notes.append(f"{source} malformed")
            continue
        x, snapped = snap(x, ctx["allowed"])
        if x is None:
            notes.append(f"{source} outside allowed options")
        elif tuple(x) in tried:
            notes.append(f"{source} already run")
        else:
            notes.append(f"{source} ok" + (" (fixed dim snapped)"
                                            if snapped else ""))
            return x, source, "; ".join(notes)
    return None, "none", "; ".join(notes)


def hypothesis_of(parsed):
    text = str((parsed or {}).get("hypothesis") or "").strip()
    return text[:200] or None


# --llm-share: the fields that travel with the token, and a length cap.
SHARED = {"none": ((), 0), "hypothesis": (("hypothesis",), 200),
          "note": (("neighbor_analysis", "hypothesis", "strategy"), 800),
          "reasoning": (("summary",), 2000)}
PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def shared_text(parsed, reasoning, mode):
    """Text sent with the token; numbers given as percentages are masked,
    so a lab's yields stay private. reasoning is the API's summary or a
    local model's full trace, of which the end (the conclusion) is sent."""
    fields, cap = SHARED[mode]
    src = dict(parsed or {}, summary=reasoning)
    text = " ".join(str(src[k]).strip() for k in fields if src.get(k))
    text = PERCENT.sub("[yield]", text)
    return (text[-cap:] if mode == "reasoning" else text[:cap]) or None
