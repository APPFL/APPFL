"""L3 prompts: one system prompt and one user prompt per decision."""

from __future__ import annotations

from math import prod

from .llm_suzuki import DIM_NAMES, REAGENT_METADATA

PEERS_SHOWN = 24

SYSTEM = (
    "You are the planning module of autonomous chemistry lab {lab} in a "
    "network of {n} labs. Every lab optimizes the yield of the same "
    "Suzuki–Miyaura coupling (heteroaryl electrophile + aryl boron "
    "nucleophile, Pd/phosphine ligand, base, solvent) and wants to find "
    "the highest-yield condition it is allowed to run in as few "
    "experiments as possible.\n\n"
    "Labs never share raw yields. After each experiment a lab sends its "
    "neighbours a token: the condition it ran (possibly perturbed for "
    "privacy) and a score q in [-1, 1], the distance of its yield from "
    "that lab's own running median in robust standard deviations, "
    "clipped at ±1 (q > 0: better than its typical result). A result "
    "marked a new best beat everything that lab had run; a token marked "
    "best so far re-sends a lab's best result.{peer_text} Labs may be "
    "restricted to different solvents, so a peer's result is evidence, "
    "not a guarantee.\n\n"
    "Most rounds a Gaussian-process acquisition function picks your "
    "lab's experiment. This round you choose, and your choice runs "
    "exactly as written. Use it where judgment beats the acquisition "
    "function: transferring a peer's pattern, testing a hypothesis that "
    "rules out a large region, or refining near a likely optimum.\n\n"
    "Reason about what your neighbours tried and why, what the evidence "
    "says about the chemistry (oxidative addition, transmetalation, "
    "boron activation, base/solvent effects), and where your lab's "
    "global optimum most likely is and how to reach it. Cite token ids. "
    "{own_text}Output strict JSON only."
)
# --llm-share: what peers' tokens carry, and what yours will carry.
PEER_TEXT = {
    "none": "",
    "hypothesis": " When a lab's planner chose the experiment, the token "
                  "also carries the planner's one-line hypothesis.",
    "note": " When a lab's planner chose the experiment, the token also "
            "carries the planner's note: analysis, hypothesis, strategy.",
    "reasoning": " When a lab's planner chose the experiment, the token "
                 "also carries an excerpt of the planner's reasoning."}
OWN_TEXT = {
    "none": "",
    "hypothesis": "Your hypothesis travels with this experiment's token; "
                  "never state numeric yields in it. ",
    "note": "Your neighbor_analysis, hypothesis and strategy travel with "
            "this experiment's token; never state numeric yields in them. ",
    "reasoning": "An excerpt of your reasoning travels with this "
                 "experiment's token; never state numeric yields. "}
SHARES = tuple(PEER_TEXT)

TASK = (
    "## Task",
    "Choose ONE condition from the allowed options that your lab has not "
    "run yet; it runs exactly as written. Give two fallbacks in case it "
    "is invalid or already run. Return JSON with these keys:",
    '{"neighbor_analysis": "<=80 words: what peers tried and why, '
    'citing token ids",',
    ' "evidence_ids": ["a2r17"],',
    ' "hypothesis": "<=200 chars: the claim this experiment tests",',
    ' "strategy": "<=60 words: how this moves your lab toward its '
    'global optimum",',
    ' "intent": "transfer | refine | test | explore",',
    ' "expected_yield": [lo, hi],',
    ' "proposal": {"electrophile_idx": 0, "nucleophile_idx": 0, '
    '"base_idx": 0, "ligand_idx": 0, "solvent_idx": 0},',
    ' "fallbacks": [<same shape as proposal>, <same shape>]}',
)


def render_system(lab, n_labs, share="hypothesis"):
    return SYSTEM.format(lab=int(lab), n=int(n_labs),
                         peer_text=PEER_TEXT[share], own_text=OWN_TEXT[share])


def name(dim, idx):
    return REAGENT_METADATA[DIM_NAMES[dim]][int(idx)]["name"]


def condition(x):
    ints = ",".join(str(int(v)) for v in x)
    names = " / ".join(name(d, v) for d, v in enumerate(x))
    return f"({ints}) {names}"


def header(ctx):
    allowed, hist = ctx["allowed"], ctx["history"]
    if ctx["setting"] == "het":
        rule = f"solvent fixed to {name(4, allowed[4][0])}"
    else:
        rule = "all options allowed"
    best = max(hist, key=lambda h: h["y"]) if hist else None
    best = (f"{best['y']:.1f}% at {condition(best['x'])}" if best
            else "none yet")
    return [f"## Lab {ctx['agent']}: {rule}",
            f"Round {ctx['round']} (of 0-{ctx['n_rounds'] - 1}). "
            f"Experiments run: {len(hist)}. Best yield so far: {best}."]


def options(allowed):
    out = ["## Allowed options (index: name, class)"]
    for d, idxs in enumerate(allowed):
        meta = REAGENT_METADATA[DIM_NAMES[d]]
        out.append(DIM_NAMES[d] + (" (FIXED):" if len(idxs) == 1 else ":"))
        out += [f"  {i}: {meta[int(i)]['name']} ({meta[int(i)]['class']})"
                for i in idxs]
    return out


def experiments(history):
    out, seen = ["## Your experiments (yields are private to your lab)"], set()
    groups = (("best", sorted(history, key=lambda h: -h["y"])[:8]),
              ("worst", sorted(history, key=lambda h: h["y"])[:5]),
              ("recent", sorted(history, key=lambda h: -h["round"])[:8]))
    for label, rows in groups:
        rows = [h for h in rows if (h["round"], tuple(h["x"])) not in seen]
        seen.update((h["round"], tuple(h["x"])) for h in rows)
        if rows:
            out.append(f"{label}:")
            out += [f"  r{h['round']} {condition(h['x'])} -> "
                    f"{float(h['y']):.1f}%" for h in rows]
    return out if len(out) > 1 else out + ["  (none yet)"]


def decisions(past):
    out = ["## Your last planned experiments"]
    for d in list(past)[-5:]:
        y = "pending" if d.get("y") is None else f"{d['y']:.1f}%"
        ints = ",".join(str(int(v)) for v in d["x"])
        out.append(f"  r{d['round']} ({ints}) -> {y}: "
                   f"\"{d.get('hypothesis') or ''}\"")
    return out if len(out) > 1 else out + ["  (none yet)"]


def _key(t):
    return t["source"], t["round"], tuple(int(v) for v in t["x"])


def shown(memory):
    """Peer tokens shown, newest first. Tag 1 re-sends a lab's best; when
    it repeats that lab's result of the round, it is folded into it."""
    new = {_key(t) for t in memory if t.get("tag") != 1}
    return [t for t in memory
            if not (t.get("tag") == 1 and _key(t) in new)][:PEERS_SHOWN]


def peers(memory, label="hypothesis"):
    out = ["## Peer tokens (newest first)"]
    resent = {_key(t) for t in memory if t.get("tag") == 1}
    for t in shown(memory):
        new_best = t.get("tag") == 2 or _key(t) in resent
        what = (f"best so far as of r{t['round']}" if t.get("tag") == 1
                else f"r{t['round']} result"
                + (" (a new best)" if new_best else ""))
        line = (f"[{t['id']}] lab {t['source']}, {what}: "
                f"{condition(t['x'])}, q={float(t['q']):+.2f}")
        if t.get("insight"):
            line += f', {label}: "{t["insight"]}"'
        out.append(line)
    return out if len(out) > 1 else out + ["(no peer tokens yet)"]


def coverage(ctx):
    allowed = ctx["allowed"]
    tried = {tuple(int(v) for v in x) for x in ctx["observed"]}
    peer = {tuple(int(v) for v in t["x"]) for t in ctx["memory"]} - tried
    least = []
    for d, idxs in enumerate(allowed):
        if len(idxs) > 1:
            n = {int(i): sum(x[d] == int(i) for x in tried) for i in idxs}
            low = sorted(n, key=lambda i: (n[i], i))[:2]
            least.append(f"{DIM_NAMES[d]} "
                         + ", ".join(f"{i} ({n[i]})" for i in low))
    return ["## Coverage",
            f"You have run {len(tried)} of {prod(len(a) for a in allowed)} "
            f"allowed conditions; peer tokens cover {len(peer)} others.",
            "Least tried by you, index (count): " + "; ".join(least) + "."]


def render_user(ctx, past=(), share="hypothesis"):
    label = share if share in ("note", "reasoning") else "hypothesis"
    blocks = (header(ctx), options(ctx["allowed"]),
              experiments(ctx["history"]), decisions(past),
              peers(ctx["memory"], label), coverage(ctx), TASK)
    return "\n\n".join("\n".join(b) for b in blocks)


def shown_ids(ctx):
    return [t["id"] for t in shown(ctx["memory"])]
