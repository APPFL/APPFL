# Quick Run Guide

Run from the repo root (this folder):

1) Default run (uses `configs/queue_case2.yaml`):

```
python main.py
```

2) Specify config explicitly:

```
python main.py --config ./configs/queue_case2.yaml
```

3) Choose algorithm (`queue_async` or `fedavg_sync`):

```
python main.py --algo fedavg_sync
```

4) Optional: override per-client local steps (single int or comma list):

```
python main.py --steps 5
python main.py --steps 10,8,6,4
```

Examples:

```
python main.py --config ./configs/queue_case2.yaml --algo fedavg_sync
python main.py --config ./configs/queue_case2.yaml --algo queue_async --steps 5
```

