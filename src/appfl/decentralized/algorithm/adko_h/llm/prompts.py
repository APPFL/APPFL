"""The campaign prompt: what one agent tells the model, and what it must not imply.

Three rules shaped this text, each from something the L3 screen got wrong:

* **the model is not asked to pick an experiment.** It is told, explicitly, that a calibrated
  rule will choose every point inside its region. The screen's prompts invited a pick and got
  confident picks that lost to the score.
* **no peer outcome appears.** An agent cannot see its neighbours' results -- only quantized
  signals, their announced regions and their settled verdicts. The earlier prompt leaked a
  shared history the method does not have, which made its advice unreproducible on the real
  problem.
* **the test must be falsifiable at the stated budget.** The model states a threshold and a
  probability before any data exists, both are scored afterwards, and the prompt says so. A
  planner that cannot be wrong cannot be measured.

The text is kept short on purpose: it is sent once per campaign per agent, and at a thousand
agents the prompt is the cost.

``SYSTEM_V2`` and :func:`user_v2` are the second version, written after the first full PMO
screen showed what the first one taught: a planner judged only on its own test optimizes that
test. Version 1 offered ``adko`` as "the default", told the planner to work elsewhere when a
neighbour had claimed a region, scaled the new-family test for campaigns of 2-10, and fed back
nothing but whether each test held. Its planners kept away from each other's productive regions,
always chose ``adko``, and wrote tests that passed 97% of the time. Version 2 defines a hit, feeds
back what each campaign found by that measure, gives the base rates a test should be set
against, describes the rules by what they trade, and presents claims as evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM = """You plan experimental campaigns for one laboratory in a federation of independent \
laboratories. Each laboratory runs its own experiments and can only exchange small signals \
with its neighbours; there is no central coordinator and no shared database of results.

Your job is NOT to choose experiments. A calibrated numeric rule chooses every experiment \
inside the region you pick. You choose, for the next few experiments:

1. region   -- which part of the design space to work in (give an interval per coordinate, or \
null to leave that coordinate free)
2. rule     -- how to choose inside it:
   - adko    : the federation's social score (own model + neighbours' signals). The default.
   - indep   : Bayesian optimization on my own data only, ignoring neighbours.
   - exploit : go to my model's predicted best.
   - explore : go where my model is least certain.
   - novelty : go furthest from everything anyone is known to have tried.
3. budget   -- how many experiments this campaign gets before it is judged.
4. hypothesis -- one sentence saying what you expect to find there, stated now.
5. test     -- the measurement that decides whether the hypothesis held: a metric, >= or <=, \
and a threshold.
6. p_success -- your honest probability that the test passes.

The test is evaluated from the data after the budget is spent, and your p_success is scored \
against what happened. A test you cannot fail is worthless. A region covering most of the \
space is not a hypothesis.

Prefer a region you can justify from the evidence you were given. If your neighbours have \
already claimed a region, working elsewhere is usually worth more than competing with them; \
if a neighbour's verdict shows a region failed, do not repeat it."""


def _rows(items: List[Dict[str, Any]], limit: int = 8) -> str:
    if not items:
        return "(nothing yet)"
    return "\n".join(
        f"- {r.get('point', '?')}  score={r.get('score')}  family={r.get('family') or '-'}"
        for r in items[:limit]
    )


def _history(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "(this is your first campaign)"
    out = []
    for c in items:
        v = c.get("verdict") or {}
        got = (
            f"{'HELD' if v.get('success') else 'FAILED'} ({v.get('metric')}={v.get('value')} "
            f"vs {v.get('op')} {v.get('threshold')})"
            if v
            else "still running"
        )
        out.append(
            f"- {c.get('region_label')} rule={c.get('rule')} budget={c.get('budget')}: "
            f"{c.get('hypothesis') or '(none)'} -> {got}"
        )
    return "\n".join(out)


def user(ctx: Dict[str, Any]) -> str:
    """The user message for one agent's campaign decision."""
    p = ctx.get("problem", {})
    metrics = ", ".join(ctx.get("metrics", []))
    first = (
        f"This is your first campaign. Your assigned starting territory is: {ctx['territory']}. "
        "Start inside it, and narrow it if you have a reason to."
        if ctx.get("first_campaign")
        else "You may work anywhere in the space."
    )
    return f"""## Task
{p.get('task', '')}

## What an outcome is
{p.get('outcome', '')}

## Design space coordinates
{ctx.get('dims', '')}

## Domain notes
{p.get('hints', '')}

## My laboratory (agent {ctx.get('agent')}, round {ctx.get('round')})
Experiments I have run: {ctx.get('n_evaluated')}
My best so far:
{_rows(ctx.get('best', []))}
Families I have found: {', '.join(ctx.get('my_families') or []) or '(none)'}
Families my neighbours reported: {', '.join(ctx.get('peer_families') or []) or '(none)'}

## My neighbours' active claims (regions they are working in now)
{ctx.get('active_claims', '(none)')}

## Settled verdicts that reached me (theirs and mine)
{ctx.get('verdicts', '(nothing yet)')}

## My previous campaigns
{_history(ctx.get('history', []))}

## Choose the next campaign
{first}
Budget must be between {ctx.get('budget_range', [2, 10])[0]} and \
{ctx.get('budget_range', [2, 10])[1]} experiments.
Available test metrics: {metrics}.
- new_families: how many family keys appear in this campaign that nobody had reported to me \
when it started (an integer; 1 is a modest claim, 3 is bold).
- novel_distance: how far this campaign's most isolated outcome lands from every outcome known \
at its start, in standardized descriptor units (1.0 is about one standard deviation).
- hull_growth: how much this campaign grows the convex hull of known outcomes.
- max_objective: the best score this campaign reaches.
Answer with the JSON object only."""


# -- version 2 ---------------------------------------------------------------------------------

_SYSTEM_V2 = """You plan experimental campaigns for one laboratory in a federation of independent \
laboratories. Each laboratory runs its own experiments and can only exchange small signals \
with its neighbours; there is no central coordinator and no shared database of results.

You do not choose experiments. A calibrated numeric rule chooses every experiment inside the \
region you pick, so the region decides what the rule can find. For the next campaign you choose:

1. region   -- an interval per coordinate, or null to leave that coordinate free
2. rule     -- how to choose inside the region:
   - adko    : my model plus my neighbours' signals; goes for high scores near what the team \
has found
   - indep   : my model alone, ignoring neighbours
   - exploit : my model's predicted best; the narrowest search
   - explore : where my model is least certain; broad coverage, few high scores
   - novelty : furthest from everything anyone is known to have run; new kinds of outcome, \
rarely high scores
3. budget   -- how many experiments this campaign gets before it is judged
4. hypothesis -- one sentence saying what you expect to find there, stated now
5. test     -- a metric, >= or <=, and a threshold, computed from the campaign's data when the \
budget is spent
6. p_success -- your probability that the test passes

The run is judged on hits: distinct families among outcomes that score at least the hit \
threshold. A region that yields many new families but no hits adds nothing to that count.

Your p_success is scored against what happens. Set the threshold from your base rates so that \
you expect the test to pass about half the time: a test that is sure to pass, or sure to fail, \
tells you nothing."""

_CLAIMS_V2 = """

Neighbours' claims show where they are working now, and their verdicts show what their \
campaigns found. Use both as evidence, as you would your own results."""

_NOTE_V2 = """

Your neighbours cannot see your results. The only account of them they get is one sentence \
you write, which travels with your claim and your verdict. Also answer with:

7. note -- one sentence to your neighbours: what your last campaign taught you about where \
hits are. Write what you would want from them: which coordinates mattered, what did not work, \
and how sure you are. It is not read by any rule, only by other planners."""

_METRICS_V2 = """- new_hit_families: families among this campaign's hits that nobody had reported to me \
when it started (an integer)
- hits: how many of this campaign's outcomes are hits (an integer)
- new_families: families in this campaign that nobody had reported to me when it started \
(an integer)
- novel_distance: how far this campaign's most isolated outcome lands from every outcome known \
at its start, in standardized descriptor units (1.0 is about one standard deviation)
- hull_growth: how much this campaign grows the convex hull of known outcomes
- max_objective: the best score this campaign reaches"""


def system_v2(ctx: Dict[str, Any]) -> str:
    """The version-2 system prompt; the claims paragraph and the note only when the arm has them."""
    out = _SYSTEM_V2 + (_CLAIMS_V2 if ctx.get("claims_on", True) else "")
    if ctx.get("claims_on", True) and ctx.get("share_detail"):
        out += _NOTE_V2
    return out


def _rates(r: Optional[Dict[str, Any]]) -> str:
    if not r:
        return "(none of my campaigns has finished yet)"
    return (
        f"over my {r['campaigns']} finished campaigns ({r['experiments']} experiments), per "
        f"experiment: {r['new_families']:.2f} new families, {r['hits']:.3f} hits, "
        f"{r['new_hit_families']:.3f} new hit families"
    )


def _recent(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "(this is your first campaign)"
    out = []
    for c in items:
        f = c.get("found") or {}
        v = c.get("verdict") or {}
        test = (
            f"test {v.get('metric')} {v.get('op')} {float(v.get('threshold', 0)):g}: "
            f"{'HELD' if v.get('success') else 'FAILED'} (value {v.get('value')})"
            if v
            else "test still running"
        )
        out.append(
            f"- {c.get('region_label')} rule={c.get('rule')} budget={c.get('budget')}: "
            f"{f.get('spent', 0)} experiments, {f.get('hits', 0)} hits, "
            f"{f.get('new_hit_families', 0)} new hit families, {f.get('new_families', 0)} new "
            f"families, best {f.get('best')}, mean {f.get('mean')}; {test}"
        )
    return "\n".join(out)


def user_v2(ctx: Dict[str, Any]) -> str:
    """The version-2 user message: the endpoint, what recent campaigns found, base rates."""
    p = ctx.get("problem", {})
    thr = ctx.get("hit_threshold")
    hit = (
        f"An outcome is a hit when its score is at least {thr:g}."
        if thr is not None
        else "No hit threshold is set."
    )
    first = (
        f"This is your first campaign. Your assigned starting territory is: {ctx['territory']}. "
        "Start inside it, and narrow it if you have a reason to."
        if ctx.get("first_campaign")
        else "You may work anywhere in the space."
    )
    claims = (
        f"""
## My neighbours' active claims (regions they are working in now)
{ctx.get('active_claims', '(none)')}

## Verdicts that reached me (theirs and mine), with what each campaign found
{ctx.get('verdicts', '(nothing yet)')}
"""
        if ctx.get("claims_on", True)
        else ""
    )
    lo, hi = ctx.get("budget_range", [2, 10])
    return f"""## Task
{p.get('task', '')}

## What an outcome is
{p.get('outcome', '')}
{hit}

## Design space coordinates
{ctx.get('dims', '')}

## Domain notes
{p.get('hints', '')}

## My laboratory (agent {ctx.get('agent')}, round {ctx.get('round')})
Experiments I have run: {ctx.get('n_evaluated')}
My best so far:
{_rows(ctx.get('best', []))}
Distinct families: I have found {ctx.get('n_families', 0)}; my neighbours have reported \
{ctx.get('n_peer_families', 0)}.
My base rates: {_rates(ctx.get('base_rates'))}.

## My recent campaigns, oldest first, with what each found
{_recent(ctx.get('recent', []))}
{claims}
## Choose the next campaign
{first}
Budget must be between {lo} and {hi} experiments.
Test metrics:
{_METRICS_V2}
Answer with the JSON object only."""
