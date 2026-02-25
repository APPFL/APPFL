# HF config template

Use `template.yaml` as a base and override exact model card id via CLI:

```bash
python -m appfl_sim.runner \
  --config appfl_sim/config/examples/external_datasets/hf/template.yaml \
  model_name=bert-base-uncased \
  experiment_name=hf-bert-test
```

This package intentionally keeps only a small template set.
Keep project-specific runs in your own config repo/folder and layer them with `--config` + CLI overrides.
