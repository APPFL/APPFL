"""The campaign schema, generated from the problem's own region space.

The planner's whole output is one object: a region, a rule, a budget, a hypothesis, a reason
and a test. Everything in it is constrained -- the rule and the metric are enumerations, the
budget is bounded, the region's coordinates are exactly the ones the problem declared -- so a
served model under guided decoding cannot answer with something the loop has to interpret.
Narrowing happens here and clipping happens in ``RegionSpace.region``; by the time a design
reaches an agent it is already legal.

The schema is built per problem rather than written out, because the coordinates differ
(molecule properties, composition fractions, Suzuki's categorical options) and a hand-written
schema would drift from the space it describes. Every coordinate is nullable and required:
``null`` is how the planner says "leave this one free", which is a meaningful choice and a
narrower schema than making properties optional, which some guided-decoding backends ignore.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from appfl.decentralized.algorithm.adko_h.campaign import BUDGET_RANGE, RULES, Test
from appfl.decentralized.algorithm.adko_h.region import KIND_CAT, RegionSpace

#: Tests the planner may write. ``novel_distance`` and ``hull_growth`` are in descriptor units,
#: which the prompt states, so a threshold is meaningful rather than guessed.
METRICS = ("new_families", "novel_distance", "hull_growth", "max_objective")
#: Version 2 adds the hit metrics, so a test can be about the endpoint itself.
METRICS_V2 = ("new_hit_families", "hits") + METRICS


def campaign_format(
    space: RegionSpace, metrics: tuple = METRICS, budget_range=None, note: bool = False
) -> Dict[str, Any]:
    """A ``json_schema`` response format for one campaign design.

    ``note`` adds the one field the detailed channel needs: a sentence addressed to the
    neighbours rather than to the agent itself. It is asked for in the same call as the design,
    so a richer channel costs output tokens and not a second request.
    """
    lo, hi = tuple(budget_range) if budget_range else BUDGET_RANGE
    props: Dict[str, Any] = {}
    for d in space.dims:
        if d.kind == KIND_CAT:
            props[d.name] = {
                "type": ["array", "null"],
                "items": {"type": "integer", "minimum": 0, "maximum": len(d.options) - 1},
                "description": f"option indices to allow for {d.name}, or null to leave free",
            }
        else:
            props[d.name] = {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "description": f"[low, high] for {d.name} within [{d.lo:g}, {d.hi:g}], or null",
            }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["region", "rule", "budget", "hypothesis", "rationale", "test", "p_success"],
        "properties": {
            "region": {
                "type": "object",
                "additionalProperties": False,
                "required": list(props),
                "properties": props,
            },
            "rule": {"type": "string", "enum": list(RULES)},
            "budget": {
                "type": "integer",
                "minimum": lo,
                "maximum": hi,
            },
            "hypothesis": {
                "type": "string",
                "description": "one sentence, stated before the data exists",
            },
            "rationale": {"type": "string", "description": "why this region and this rule"},
            "test": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric", "op", "threshold"],
                "properties": {
                    "metric": {"type": "string", "enum": list(metrics)},
                    "op": {"type": "string", "enum": [">=", "<="]},
                    "threshold": {"type": "number"},
                },
            },
            "p_success": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "your probability that the test passes",
            },
        },
    }
    if note:
        schema["required"].append("note")
        schema["properties"]["note"] = {
            "type": "string",
            "description": (
                "one sentence to your neighbours: what your last campaign taught you about "
                "where hits are, in your own words"
            ),
        }
    return {"type": "json_schema", "name": "campaign", "schema": schema, "strict": True}


def parse(space: RegionSpace, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A model's answer turned into a design, or ``None`` if there is nothing usable.

    Forgiving about shape and strict about meaning: a malformed interval is dropped (leaving
    that coordinate free), an unknown rule becomes the settled one, and a budget outside the
    range is clipped -- but a reply with no region and no test at all is refused, so a broken
    call is counted as an error instead of quietly becoming a random campaign.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("region")
    if not isinstance(raw, dict):
        raw = {}
    bounds: Dict[str, Any] = {}
    picks: Dict[str, Any] = {}
    for d in space.dims:
        v = raw.get(d.name)
        if v is None or not isinstance(v, (list, tuple)):
            continue
        if d.kind == KIND_CAT:
            picks[d.name] = [int(i) for i in v if isinstance(i, (int, float))]
        elif len(v) >= 2:
            bounds[d.name] = [float(v[0]), float(v[1])]
    t = data.get("test") if isinstance(data.get("test"), dict) else {}
    if not bounds and not picks and not t:
        return None
    try:
        test = Test(
            metric=str(t.get("metric", "new_families")),
            op=str(t.get("op", ">=")),
            threshold=float(t.get("threshold", 1.0)),
        )
    except (ValueError, TypeError):
        test = Test()
    return {
        "region": space.region(bounds=bounds, picks=picks),
        "rule": str(data.get("rule", "adko")),
        "budget": int(data.get("budget", 5)) if str(data.get("budget", "")).strip() != "" else 5,
        "hypothesis": str(data.get("hypothesis", "")),
        "rationale": str(data.get("rationale", "")),
        "note": str(data.get("note", "")),
        "test": test,
        "p_success": float(data.get("p_success", 0.5)),
        "source": "llm",
    }
