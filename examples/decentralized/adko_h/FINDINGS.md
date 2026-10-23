# ADKO-H findings

Newest first. One entry per finding: date, what was run (oracle, arms, seeds, budget, model),
the number, and what it means. Numbers come from `report.py` or `run.json`, never from memory.

## 2026-10-23 — batches 1–2: the first planner's regions were a prior; feedback about hits makes it plan

`zaleplon_mpo` with the first screen's settings (10 agents × 1,000 rounds, random 4-regular
graph, seeds 42–49, campaigns of 20–60; the two region controls use 30, as that planner chose).
Code `4c03c879`. A rerun of `adko` with it reproduced all 8 seeds of the first screen record for
record. Planner arms through MAG: 40 runs, 0 errors, 0 fallbacks.

| Arm | Regions from | diverse hits | vs `adko` (p, every seed) | AUC top-10 | $/run |
|---|---|---|---|---|---|
| `adko` | — | 186.1 | – | 0.472 | – |
| `adko-skip` | — | 203.3 | +17.1 (0.195, no) | 0.469 | – |
| `adko-noise` | — | 201.6 | +15.5 (0.078, no) | 0.472 | – |
| `adko-noise-skip` | — | 198.8 | +12.6 (0.070, no) | 0.469 | – |
| `random-region` | random boxes on pool molecules, the planner's size | 171.3 | −14.9 (0.086, no) | 0.468 | – |
| `replay-regions` | the first `adkoh-noclaims` planner's regions, replayed | 229.1 | +43.0 (0.008, yes) | 0.476 | – |
| `adkoh`, Haiku | `claude-haiku-5-5@amazon`, prompt v1 | 126.4 | −59.8 (0.008, worse) | 0.464 | 0.24 |
| `adkoh-noclaims`, Haiku | the same, claims off | 215.0 | +28.9 (0.016, no) | 0.472 | 0.20 |
| `adkoh2-noclaims` | `claude-sonnet-5-5-medium@amazon`, prompt v2 | 250.0 | +63.9 (0.016, no) | 0.474 | 2.35 |
| `adkoh2-noclaims-blind` | the same, objective not named | 308.0 | +121.9 (0.008, yes) | 0.482 (yes) | 2.32 |
| `adkoh2` | prompt v2, claims carry findings, no claim penalty | **336.0** | +149.9 (0.008, yes) | **0.489** (yes) | 3.63 |

For comparison, the first screen: `adkoh-noclaims` 227.0, `adkoh` 147.3, `adko-div` 198.8.

- **The first planner's gain was where its regions sit, not planning.** Replayed in random order
  with no model, its regions give 229.1 against the planner's own 227.0 (+2.1, p = 0.63). Both
  beat random boxes of the same size at every seed (+57.9, +55.8). The rule's yield inside a
  region is the same for random and planner boxes (about 2.5×); what differs is the hit share
  inside them (0.91% against 1.36%; the pool's is 0.80%).
- **Prompt v2 makes the planner learn during a run.** v2 defines a hit, feeds back each
  campaign's hits, gives the agent's base rates, adds the tests `hits` and `new_hit_families`, lists
  the rules without a default and drops the instruction to work elsewhere. Region quality under v1
  is flat from the first campaign to the last (1.3–1.4%). Under v2 it rises to 1.6–1.7% (named,
  no claims), 2.0% (blind) and 3.1% (claims carrying findings). v2 agents keep a region after a
  campaign with 3 or more hits (overlap with the previous region 0.94) and leave after none
  (0.63); v1 agents moved every time (0.07).
- **Claims help once they carry findings.** In `adkoh2`, 24% of campaigns move into a region where
  a neighbour just found 3 or more hits, and they hit 7.0% of the time (other moves 3.7%, staying
  7.4%). Without verdicts, v2 agents almost never make that move (2% and 0.4% of campaigns).
  `adkoh2` beats `adkoh2-noclaims` by +86.0 at every seed.
- **Naming the objective anchors the planner.** The named v2 planner centres regions at logP 2.5
  (IQR 2.5–2.75), near zaleplon's own properties, which it cites. The pool's hits sit at logP 3.6
  (3.1–4.1). Blind regions centre at 3.25, and blind beats named by +58.0 at every seed. Named and
  without claims, v2 gains only +23.0 over v1 (6 of 8 seeds, p = 0.15).
- **v2's tests carry information.** 53–58% pass, against 96–99.5% under v1. Brier skill against
  the base rate is about 0 (−0.02 to +0.01): calibrated, not yet discriminating.
- **Haiku with prompt v1** picks richer regions than Sonnet when claims are off (hit share 1.93%
  against 1.37%), but runs 38% of those campaigns under `explore` or `novelty`, which hit about 1%
  against 4.1% under `adko`. Net −12.0 against Sonnet without claims (not significant) and −20.9
  with claims (every seed), at $0.20–0.24 a run.
- **Skipping reported inputs** sends 412 (exact match) or 616 (with noise, radius 0.30) more of the
  10,000 evaluations to new molecules, at every seed; diverse hits +12.6 to +17.1, not
  significant. Token noise at v2's level (0.173 of each coordinate's spread) costs nothing
  measurable. It defeats an exact-match skip but not a radius one: its 99% displacement (0.27) is
  far below the distance to the nearest other molecule (median 0.82). The v2 arms repeat more
  (7,770–8,232 distinct molecules against 8,777); skipping there is untested.
- **Cost.** $69.91 for the 40 MAG runs, pilot included. Prompt v2 is cheaper per run than v1
  ($2.32–3.63 against $4.21–5.12) because it drops the two lists of 20 scaffold SMILES.
- **Not yet shown:** which part of v2 does the work; claims with findings and a blind objective
  together; any other oracle.

## 2026-10-23 — first full screen with a MAG planner: claims cost hits

`zaleplon_mpo`, 10 agents × 1,000 rounds (10,000 evaluations), random 4-regular graph, seeds
42–49, campaigns of 20–60 in every arm. The four `adkoh*` arms planned with
`claude-sonnet-5-5-medium@amazon`; the other three use no model. Every model run planned with
the model: 10,475 designs, 0 errors, 0 fallbacks, no truncated replies.

| Arm | diverse hits | #Circles | hull | AUC top-10 | $/run |
|---|---|---|---|---|---|
| `adko` | 186.1 | 212.0 | 159.3 | 0.472 | – |
| `adko-div` | 198.8 | 216.8 | 166.3 | 0.467 | – |
| `random-campaign` | 76.8 | **274.0** | **213.1** | 0.449 | – |
| `adkoh` | 147.3 | 217.9 | 164.3 | 0.465 | 5.12 |
| `adkoh-noclaims` | **227.0** | 208.1 | 151.8 | **0.477** | 4.21 |
| `adkoh-noterritory` | 149.5 | 221.9 | 167.5 | 0.470 | 4.45 |
| `adkoh-nofootprint` | 162.9 | 215.6 | 159.5 | 0.466 | 4.53 |

- **The method as specified does not pass.** `adkoh` loses to `adko` on diverse hits at every
  seed (−38.9, p = 0.008) and gains nothing significant on #Circles, hull or AUC.
- **Without claims and verdicts it does.** `adkoh-noclaims` beats `adko`, `adko-div`,
  `random-campaign` and `adkoh` on diverse hits at every seed (+40.9, +28.3, +150.3, +79.8;
  p = 0.008 each, the floor for 8 paired seeds), with AUC no worse. It adds no coverage
  (#Circles, hull). So the claim channel is what costs hits here. How much of the gain is the
  model's regions is not yet separated: the random control keeps its claims and draws its rule
  at random (about 80% of its campaigns use a rule other than `adko`). A control with random
  regions, rule `adko` and claims off would separate it.
- **The planner barely uses its levers.** 98.9% of the designs chose rule `adko` (101
  `novelty`, 11 `exploit`, 1 `explore`), and every budget was 30 or 40. The model contributes
  the region and the test, not the rule, so the rule headroom measured on 2026-10-19 goes unused.
- **Its tests are too easy.** 96–99.5% pass, and Brier skill against its own base rate is
  negative in every arm (−1.5 to −7.1).
- **Cost and speed.** $146.45 for 32 runs (a one-seed pilot included). Claims and verdicts
  raise bits per token from ~530 (`adko`) to 1,150–1,420; `noclaims` sends 630. 31 runs, 8 at a
  time on one node, took 1 h 42 min at ~100 calls a minute: the per-user rate limit, not CPU,
  set the pace (one run alone takes ~13 min).

## 2026-10-23 — MAG planner, first calls

- `claude-haiku-5-5@amazon`: ~$0.0003 and ~3 s per campaign call; designs parse.
- `claude-sonnet-5-5-medium@amazon`: ~$0.014 and ~4 s per call on the PMO prompt (~4.5k input
  tokens, ~370 output); 12/12 usable. Its pilot campaigns passed 8/8 of their own tests —
  watch whether the tests are too easy.
- `claude-opus-5-5-high@amazon`: `finish_reason=content_filter` with empty text on 11/11 PMO
  campaign calls, billed ~$0.02 each. The 20% failure guard stopped the run. Avoid this route.
- The first real-server run exposed a bug the mock planner could not: a new event loop per
  round broke the HTTP client after round one ("Event loop is closed"). Fixed; regression test
  added. This would have affected vLLM runs equally.

## 2026-10-19 — rule headroom on PMO

`zaleplon_mpo`, 10 agents, 200 rounds (2,000 evaluations), random 4-regular graph, seeds 0–1,
no planner, rule fixed per arm (`--rule`):

| Rule | diverse hits | #Circles | hull | AUC top-10 |
|---|---|---|---|---|
| `adko` | 36.0 | 141.0 | 108.5 | 0.422 |
| `indep` | 30.0 | 158.5 | 128.4 | 0.409 |
| `exploit` | 31.5 | 154.0 | 119.5 | 0.418 |
| `explore` | 5.5 | 202.0 | 173.8 | 0.355 |
| `novelty` | 7.5 | 198.0 | 163.5 | 0.382 |
| `adko` + novelty bonus (w=1) | 35.0 | 137.0 | 105.8 | 0.412 |

Exploration buys coverage at about five times fewer hits; no fixed rule does both. The order
within each group flipped between the two seeds. At weight 1 the bonus does not add coverage.

Also: one full-budget run (10,000 evaluations, mock planner, seed 0) took 146 s: 6,735
scaffolds, 87 diverse hits at 0.4, 283 #Circles, hull 211.4, AUC-top-10 0.459, 490 campaigns
(269 passed), Brier 0.271. Making peer footprints outlive the token FIFO left the PMO `novelty`
runs bit-identical (with 250k candidates the max-min pick is dominated by the pool's geometry).
