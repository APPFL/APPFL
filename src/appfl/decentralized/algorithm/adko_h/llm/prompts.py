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
"""

from __future__ import annotations

from typing import Any, Dict, List

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
