# Algorithm Placeholders

This folder provides placeholder simulation configs for algorithms implemented in
`appfl_sim/algorithm/{aggregator,scheduler,trainer}`.

Notes:
- These are intended as starter configs for users.
- Current simulator path uses APPFL-style `ServerAgent` and `ClientAgent`.
- New algorithm wiring is convention-driven from `algorithm.name`:
  `<PascalCase(name)>Aggregator`, `<PascalCase(name)>Scheduler`, `<PascalCase(name)>Trainer`.
- Optional explicit overrides (`algorithm.aggregator`, `algorithm.scheduler`, `algorithm.trainer`)
  should still follow the same class naming prefix for clarity/reuse.
- Users do not need to edit `runner.py` when adding new algorithms that follow this convention.
- Evaluation-focused examples are under `appfl_sim/config/algorithms/evaluation/`.
