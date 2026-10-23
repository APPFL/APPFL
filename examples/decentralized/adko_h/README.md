# ADKO-H on a Slurm cluster: ready to run

ADKO-H is ADKO v2 with the language model moved up one level. It no longer picks experiments —
the L3 screen measured that and it lost to the calibrated score. It picks **campaigns**: a
region of the design space, one of five numeric rules, a budget, and a test it can fail. Every
experiment inside a campaign is still the rule's argmax. When a campaign opens, its region is
announced to the agent's neighbours as a **claim** (they steer away softly, and it expires);
when it closes, its **verdict** is computed from the data and relayed several hops. No
allocator, no central board — which is the part that has to survive at hundreds of agents.

Code: `src/appfl/decentralized/algorithm/adko_h/`. Tests: `tests/decentralized/test_adko_h.py`.

The proxy problem is **PMO** (Gao et al., NeurIPS 2022 D&B): 250k purchasable molecules, 20
goal-directed oracles, a published call budget, and an established diversity literature
(#Circles, diverse hits). It is CPU-only and cheap, so the GPU is needed only to serve the
model that plans.

## Run it

```bash
# 0. the code
git clone https://github.com/APPFL/APPFL.git && cd APPFL && git checkout daisy/adko-h
export ADKO_H=$PWD
export ADKO_DATA=$PWD/../adko_h_data          # writable, outside the repo
export ADKO_RESULTS=$ADKO_DATA/results

# 1. ONCE, on a login node (compute nodes have no internet)
bash examples/decentralized/adko_h/setup.sh
#   creates conda env "adko_h" (numpy, scipy, rdkit, openai — no torch, about a minute)
#   downloads ZINC-250k, builds the fingerprint and embedding caches, runs the tests

conda activate adko_h
export PYTHONPATH=$ADKO_H/src:$PYTHONPATH
export ADKO_PMO_POOL=$ADKO_DATA/zinc250k.smi

# 2. the arms that need no GPU (one array per arm, one seed per array task)
for ARM in adko adko-div random-campaign; do
  ARM=$ARM ORACLE=zaleplon_mpo sbatch -p <cpu-partition> \
    examples/decentralized/adko_h/screen.sbatch
done

# 3. the model, then the arms that use it
sbatch -p <gpu-partition> examples/decentralized/adko_h/serve.sbatch
for ARM in adkoh adkoh-noclaims adkoh-noterritory adkoh-nofootprint; do
  ARM=$ARM ORACLE=zaleplon_mpo sbatch -p <cpu-partition> \
    examples/decentralized/adko_h/screen.sbatch
done
#   the run jobs wait for $ADKO_DATA/llm_endpoint.txt, so submission order does not matter

# 4. read it
python examples/decentralized/adko_h/report.py $ADKO_RESULTS/zaleplon_mpo
```

Defaults per run: 10 agents, 1000 rounds — **10,000 oracle calls, PMO's standard budget** — a
random 4-regular graph, a 256-molecule pool per agent-round, campaigns of 20–60 experiments,
seeds 42–49. One arm, one seed is one array task.

**Planning through the AmSC gateway (MAG) instead of a served model** — no GPU, paid per call,
rate-limited per user. All MAG arms go in one job on one node, with an explicit dollar cap:

```bash
export AMSC_TOKEN_FILE=/path/to/token          # chmod 600, outside any repo
python -m appfl.decentralized.algorithm.adko_h.llm.mag ping --model claude-haiku-5-5@amazon
MODEL=claude-sonnet-5-5-medium@amazon MAG_BUDGET=10 ARMS=adkoh SEEDS=42 \
  sbatch -p <cpu-partition> examples/decentralized/adko_h/screen_mag.sbatch
```

`AGENTS.md` has the limits, measured costs per model, and the models that did not work.

A smoke test first, which needs nothing but the environment:

```bash
python -m appfl.decentralized.algorithm.adko_h.run --problem synthetic --arm adkoh \
    --planner mock --rounds 14 --agents 4 --seeds 0
```

## The arms

| Arm | Planner | What it answers |
|---|---|---|
| `adko` | none | ADKO v2's settled method over a sampled pool. The baseline everything is paired against. |
| `adko-div` | none | …plus a novelty bonus. **The arm the LLM has to beat**: diversity pressure without tokens. |
| `random-campaign` | random | The campaign lifecycle with random designs. Separates the structure from the planning. |
| `random-region` | random | Claims off, random boxes placed where candidates are and sized like a planner's. Are regions alone worth it? |
| `replay-regions` | replay | Claims off, a planner arm's own regions in random order. A fixed prior, or planning? |
| `adkoh` | served model | The method. |
| `adkoh-noclaims` | served model | Claims and verdicts off. Is the coordination doing the work? |
| `adkoh-noterritory` | served model | No starting Voronoi cell. Do agents divide the space by message alone? |
| `adkoh-nofootprint` | served model | Tokens carry no descriptor or family key. Is the shared footprint what prevents duplication? |
| `adkoh2`, `adkoh2-noclaims` | served model | Prompt v2: the planner is told what a hit is and what each campaign found. Can it then use its levers? |
| `adkoh2-noclaims-blind` | served model | Prompt v2 without the objective's name. What is knowing the benchmark worth? |
| `adkoh3` | served model | The parts that paid, composed: prompt v2, claims carrying findings, no claim penalty, the objective named, v2's noisy input channel. The method as it would ship. |
| `adkoh3-blind` | served model | …without the objective's name. Composed, what is knowing the benchmark worth? |
| `adkoh3-nofindings` | served model | Claims and verdicts still travel, but a verdict reports only whether its own test held. Is it the findings or the channel? |
| `adkoh3-penalty` | served model | The same claims with v1's steer-away penalty back. Does the fence still cost hits once claims carry evidence? |
| `adkoh3-rich` | served model | The detailed channel: the full hypothesis, a sentence the planner writes for its neighbours, and the box its hits fell in. Is prose worth its bits? |

Other problems and graphs are flags, not scripts: `--problem synthetic` (free: continuous
inputs, a descriptor and family keys, like an expensive simulator would give), `--topology random_regular --degree 4` (or `ring`, `fully_connected`,
`random_geometric`), `--agents 100`.

## What it costs (measured on one node, 2 BLAS threads)

| | |
|---|---|
| 1,000 evaluations (100 rounds × 10 agents), 20k pool | **3.5 s** |
| 10,000 evaluations — one full-budget arm, one seed, 250k pool | **146 s** (measured, mock planner) |
| Pool preparation, once | ≈ 40 s fingerprints + ≈ 4.5 min descriptors and projection, then cached (71 MB) |
| Model calls per full-budget run, campaigns of 20–60 | **490** (campaigns of 2–10 would be ≈ 2,000) |
| 7 arms × 8 seeds, CPU part | **≈ 3 CPU-hours total**, trivially parallel across array tasks |
| 4 LLM arms × 8 seeds | **≈ 2–4 A100-hours**, almost all of it the served model |

So a PMO screen is a GPU-afternoon, not a GPU-week.

One full-budget reference run (`zaleplon_mpo`, seed 0, mock planner, 10 agents, 4-regular):
6,735 distinct scaffolds, 87 diverse hits at 0.4, 283 #Circles, hull 211.4, AUC-top-10 0.459,
best 0.545, 490 campaigns of which 269 passed their own test, Brier 0.271.

## Known limits, stated rather than discovered later

- **The shared embedding is a 32-dimensional random projection of the ECFP4 fingerprint**, not
  the fingerprint. A token has to stay small and ADKO's GP, kernel and token all read one
  vector. Every arm pays the same cost, so arm-to-arm comparisons are unaffected, but absolute
  AUC sits below the published PMO leaderboard. Do not compare these AUCs to the leaderboard.
- **The campaign channel is not free.** A bare v2 token is ~150 bits; with claims and verdicts
  relayed 3 hops on a 4-agent graph it measured ~1,600–1,800 bits. On a small graph nearly
  everyone has already heard the message, so that ratio is the worst case — it should fall at
  100 agents, and `report.py` prints it per arm. Worth checking before claiming the channel is
  cheap.
- **No mid-run resume.** A killed job is re-run; the model calls replay from the SQLite cache
  for free, the oracle calls are seconds. For an expensive problem, resume belongs in the
  evaluator.
- **Three PMO oracles are missing** (drd2, gsk3b, jnk3): they need downloaded sklearn pickles.
  The other 20 are vendored from PyTDC (MIT) and tested bit-identical to it.
- **PMO is a benchmark, not an application.** It exercises the regime ADKO-H is for — a space too
  large to enumerate, qualitative families, a descriptor space — and any application with that
  shape plugs in through `adko_h/problem.py` (pool, batch evaluate, descriptor, family key).
