# ADKO-H findings

Newest first. One entry per finding: date, what was run (oracle, arms, seeds, budget, model),
the number, and what it means. Numbers come from `report.py` or `run.json`, never from memory.

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
