import argparse
from omegaconf import OmegaConf
from builders.appfl_builders import build_server, build_clients
from algorithms.queue_async import QueueAsyncFL
from algorithms.fedavg_sync import FedAvgSync
from utils.seeds import set_seeds

ALGO_REGISTRY = {
    "queue_async": QueueAsyncFL,
    "fedavg_sync": FedAvgSync,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./configs/queue_case2.yaml")
    ap.add_argument("--algo", type=str, default=None, help="queue_async | fedavg_sync (overrides YAML)")
    ap.add_argument("--steps", type=str, default=None,
                    help="Optional per-client local steps (single int or comma list). Overrides YAML for the chosen algorithm.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    algo_name = args.algo or str(getattr(getattr(cfg, "algo", {}), "name", "queue_async"))
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo '{algo_name}'. Choose from: {list(ALGO_REGISTRY.keys())}")

    set_seeds(int(cfg.system.seed))

    server = build_server(cfg)
    clients, client_ids = build_clients(cfg)

    AlgoCls = ALGO_REGISTRY[algo_name]
    runner = AlgoCls(cfg=cfg, server=server, clients=clients, client_ids=client_ids, override_steps=args.steps)
    runner.run()

if __name__ == "__main__":
    main()
