# Suzuki ADKO Reproduction

This example runs the Suzuki-Miyaura ADKO experiment from
`lucasrillo/adko/scientific_discovery` on APPFL's decentralized runtime. The
current scripts cover the LM-off ADKO arm and compare APPFL outputs against the
reference JSON results.

## Setup

Clone the reference repo under APPFL's ignored `.external/` directory:

```bash
cd APPFL
mkdir -p .external
git clone https://github.com/lucasrillo/adko.git .external/adko_ref
```

The scripts expect:

```text
.external/adko_ref/scientific_discovery
```

Then follow `.external/adko_ref/scientific_discovery/README.md` to install the
reference dependencies. That setup provides the Suzuki dataset, warmup files,
and published result JSONs used for comparison.

## Run

Run the APPFL LM-off sweep:

```bash
examples/decentralized/suzuki/run_appfl_suzuki_main_llmoff.sh
```

Outputs are written to `results/appfl_suzuki_main_llmoff/`.

```bash
# quick subset
SEEDS=3 PARALLEL=1 examples/decentralized/suzuki/run_appfl_suzuki_main_llmoff.sh

# deterministic token-noise check
P_NOISE=0 examples/decentralized/suzuki/run_appfl_suzuki_main_llmoff.sh
```

## Compare

```bash
examples/decentralized/suzuki/compare_appfl_suzuki_main_llmoff.sh
```

The comparison writes:

```text
results/appfl_suzuki_main_llmoff/figures/
results/appfl_suzuki_main_llmoff/reports/
```

```bash
# include first per-step mismatch in the report
STRICT_STEPS=1 examples/decentralized/suzuki/compare_appfl_suzuki_main_llmoff.sh
```

`STRICT_STEPS=1` adds an exact step-by-step check. It reports the first seed,
round, agent, and field where APPFL differs from the reference for values such
as selected reaction `theta_int`, observed yield `y_raw`, peer-score terms, and
token-memory counts. Use it for debugging implementation differences; aggregate
metrics are usually the right evidence for reproduction.

## How to Read Results

- One seed is one full run with 200 rounds.
- One round has one evaluation per agent.
- `system_best[t]` is the best yield found by any agent up to round `t`.
- The mean `system_best` curve averages `system_best[t]` across seeds.
- The AUC-style summary averages `system_best` over rounds within each seed,
  then compares APPFL and reference paired by seed.
