# AGENTS.md — running and analysing ADKO-H

For an agent (or a person) running ADKO-H experiments on a cluster. Read this, then `README.md`
(the commands). Record what you learn in `FINDINGS.md`, newest first, with the numbers taken
from `report.py` or `run.json`.

## Rules

- **Never print, log, copy or commit a credential.** The MAG planner reads `AMSC_TOKEN` or the
  file named by `AMSC_TOKEN_FILE` (chmod 600, outside any repository). Do not `echo`, `cat` or
  `env | grep` it.
- **This repository is public.** Commit nothing that names private projects, private
  repositories, people, machines or paths. Scripts take `ADKO_H`, `ADKO_DATA`, `ADKO_RESULTS`,
  `ADKO_PMO_POOL`; keep it that way. Push only when the person running the study asks, and only
  to the branch they name.
- **Ask before** submitting more than a pilot, and before any MAG spend beyond the cap you were
  given.
- **Pilot first:** one seed of an arm before eight. A broken arm then costs one task.

## What ADKO-H is

ADKO v2 (decentralized Bayesian optimization: private GPs, one quantized knowledge token per
neighbour per round, a calibrated social score) with a language model one level up. The model
does not pick experiments; it plans **campaigns**: a region, one of five rules (`adko`, `indep`,
`exploit`, `explore`, `novelty`), a budget, a hypothesis, and a test it can fail. The rule's
argmax inside the region picks every experiment. Opening a campaign announces a **claim**
(neighbours steer away softly; claims expire); closing it relays a **verdict** computed from the
data, up to 3 hops. The aim is more *distinct* outcomes per evaluation, with no central
coordinator, on sparse graphs of many agents.

Code: `src/appfl/decentralized/algorithm/adko_h/`. Tests (offline, about a minute):
`python -m unittest discover -s tests/decentralized -t .` — set `ADKO_PMO_POOL` to include the
molecule tests.

## Running

`README.md` has the sequence: `setup.sh` once on a login node; one `screen.sbatch` array per arm
(`ARM=... ORACLE=...`); `serve.sbatch` for arms planned by a served model; `screen_mag.sbatch`
for arms planned through MAG; `report.py` to read results.

| Arm | What it answers |
|---|---|
| `adko` | the baseline every other arm is paired against |
| `adko-div` | adds a novelty bonus without a model (weight `AgentConfig.novelty_weight`) |
| `random-campaign` | same campaign lifecycle, random designs — separates structure from planning |
| `random-region` | claims off, random boxes (`--random-center pool --random-widths ...` puts them where candidates are, sized like a planner's) — are any regions worth it? |
| `replay-regions` | claims off, a planner arm's own regions replayed in random order (`--replay-from`) — a fixed prior, or planning? |
| `adkoh` | the method |
| `adkoh-noclaims`, `-noterritory`, `-nofootprint` | ablations of coordination, starting territories, shared outcome footprint |
| `adkoh2`, `adkoh2-noclaims` | prompt v2: hits defined, what each campaign found, base rates, no default rule; claims as evidence (no penalty) |
| `adkoh2-noclaims-blind` | prompt v2 without naming the objective — what knowing the benchmark is worth |
| `adkoh3` | the parts that paid, composed: prompt v2, claims carrying findings, no penalty, the objective named, v2's noisy input channel (in the arm, so no flag) |
| `adkoh3-blind`, `-nofindings`, `-penalty`, `-rich` | one part off each: the objective unnamed, verdicts without findings, the steer-away penalty back, the detailed channel on. `adkoh3` minus the noise is `adkoh2`, so that ablation needs no arm. |

`--rule <name>` fixes the rule of every campaign. On a no-planner arm, how far the fixed-rule
arms land apart is the headroom a campaign planner has; measure it on a new problem before
spending on a model. On a planner arm it keeps the planner's regions and takes the rule out.

Two switches apply to any arm. `--skip-peer-inputs` leaves out candidates a neighbour's token
shows were already run (agents otherwise re-run each other's experiments). `--token-noise`
adds Gaussian noise to the input a token carries, in units of each coordinate's spread (v2's
continuous channel is 0.17); with it, give `--skip-radius` a distance that covers the noise,
since an exact match no longer happens. `--share-detail` widens the campaign channel to prose:
the full hypothesis instead of a truncation, one sentence the planner is asked to write for its
neighbours (a `note` field in the same call, so it costs output tokens and not a second
request), and the box a campaign's hits fell in. It raises bits per token by a few hundred,
which the run record reports, so the question it asks is whether the words earn them.

## MAG (the AmSC Model Access Gateway)

OpenAI-compatible, LiteLLM underneath; the URL defaults to the public gateway and can be
overridden with `AMSC_MAG_URL`. Planner: `--planner mag --llm-model <id>`.

From the node that will run the job, before anything else:

```bash
python -m appfl.decentralized.algorithm.adko_h.llm.mag ping --model claude-haiku-5-5@amazon
python -m appfl.decentralized.algorithm.adko_h.llm.mag spend
```

If `ping` fails from a compute node, outbound traffic is probably blocked there. Report it; do
not improvise tunnels, proxies or long jobs on a login node.

- **Limits are per user**, across everything that user runs: 20 requests in flight, 120 a minute
  (as of October 2026). The client stays under them with slot files (`MAG_SLOTS=18`,
  `MAG_GAP=10`). Slot files are local, so **run all MAG arms in one job on one node** —
  `screen_mag.sbatch` does — or divide `MAG_SLOTS` between nodes.
- **Money is capped by a ledger you set:** `MAG_LEDGER` (file) and `MAG_BUDGET` (dollars) are
  required. At the cap the run stops with `BudgetSpent`; it never continues on random designs.
- `mag spend` prints what the gateway reports (spend, personal and team budgets). Treat it as
  information; the ledger cap is the rule.
- Effort is part of the model id (`claude-sonnet-5-5-medium@amazon`). The client drops
  vLLM-only fields and sends no temperature unless asked; `--llm-max-tokens` defaults to 16,000.

Measured per campaign call on the PMO prompt (about 4.5k input tokens), October 2026:

| Model id | $ / call | s / call | Notes |
|---|---|---|---|
| `claude-haiku-5-5@amazon` | ~0.0003 | ~3 | works |
| `claude-sonnet-5-5-medium@amazon` | ~0.014 | ~4 | works |
| `claude-opus-5-5-high@amazon` | ~0.02 | ~2 | returned `finish_reason=content_filter` with no text on every call, still billed — do not use this route for this prompt |

A full-budget PMO run plans about 250–500 campaigns at campaign lengths 20–60, so roughly
$4–7 per run with Sonnet-medium. Measure again with a one-seed pilot; `report.py` prints `$/run`.

If a model returns `content_filter`, empty text or refusals, stop using it, record it in
`FINDINGS.md`, and switch model or provider route. Do not rewrite the prompt to get past a
provider's filter. The planner stops a run on its own when more than 20% of designs fail.

## Reading results

Each run writes `$ADKO_RESULTS/<oracle>/pmo-<oracle>/<arm>/seed<N>/`:

| File | What |
|---|---|
| `run.json` | config, topology, endpoints, campaign summary, planner totals, per-agent traces |
| `rounds.jsonl` | team families, hull, best, model calls, claims and verdicts per round |
| `records.csv` | one row per experiment: round, agent, key, molecule, score, family, rule, campaign |
| `llm_calls.jsonl` | every planner call: usage, cost, latency, status, parsed design, start of text |
| `llm_cache.sqlite` | replay cache: re-running a seed costs no tokens |

`python examples/decentralized/adko_h/report.py $ADKO_RESULTS/<oracle> [--baseline adko]` pairs
each arm with the baseline by seed (paired Wilcoxon) and prints, per endpoint, the mean, the
paired difference, p, and **every seed** — whether the arm beat the baseline at every seed. A
win is a better mean *and* "yes" there.

Check, in this order:

1. **Did the arm run as intended?** `fallback` should be about 0; a planner arm with many
   fallbacks is the random control. `jq -c 'select(.error or .design.source != "llm")' llm_calls.jsonl`.
2. **Endpoints:** `diverse_hits` and `n_circles` (diversity), `hull_volume` (coverage of
   outcome space), `auc_top10` (that diversity did not cost the objective).
3. **Baselines:** `adkoh` has to beat `adko`, `adko-div` *and* `random-campaign`. Beating only
   `adko` shows the structure helps, not the model.
4. **Are the tests real?** `pass` near 100% means the planner writes tests it cannot fail;
   `skill` above 0 (Brier against its own base rate) means its probabilities discriminate.
5. **Cost:** `$/run`, tokens, `bits/tok` (claims and verdicts cost several times a bare
   ~150-bit token on small graphs).
6. **Balance:** `endpoints.load_balance.gini` in `run.json`.

```bash
# campaigns: region, rule, budget, test, stated probability, outcome
jq -c '.agents[].campaigns[] | {a: .claim_id, region: .region_label, rule, budget, test: .test, p: .p_success, ok: .verdict.success}' run.json | head
# picks whose campaign region held no pool candidates (the rule then used the whole pool)
jq '[.agents[].records[] | select(.d_region_empty == true)] | length' run.json
# MAG spend recorded by this study's runs
awk '{s+=$1} END {printf "$%.2f\n", s}' "$MAG_LEDGER"
```

## Known limitations

- PMO's shared embedding is a 32-dimensional random projection of the fingerprint. Arms compare
  with each other; absolute AUC does not compare with the PMO leaderboard.
- Territory boxes are drawn in the unit cube and ignore constraints between inputs (such as
  fractions summing to at most one). An infeasible box makes the pool fall back to the whole
  space and sets `d_region_empty`. A problem with such constraints should supply feasible
  territory centres.
- No mid-run resume; a killed run restarts, and planner calls replay from the cache for free.
- Three PMO oracles are excluded (drd2, gsk3b, jnk3), which need downloaded sklearn models.
