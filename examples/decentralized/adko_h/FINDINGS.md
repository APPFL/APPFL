# ADKO-H findings

Newest first. One entry per finding: date, what was run (oracle, arms, seeds, budget, model),
the number, and what it means. Numbers come from `report.py` or `run.json`, never from memory.

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
