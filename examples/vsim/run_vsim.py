"""Entry point for the virtual-time asynchronous FL simulator (v1).

Async counterpart of examples/serial/run_serial.py. Reuses APPFL ServerAgent /
ClientAgent and drives them with AsyncSimDriver (virtual-time event queue).

Run from the examples/ directory (relative ./resources paths in the config):
    cd examples
    python vsim/run_vsim.py --num_clients 2 --seed 42
"""

import argparse
import random
from datetime import datetime

from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.vsim import AsyncSimDriver, ClientProfile, SyncSimDriver, VsimLogger


def build_profiles_from_json(client_ids, path):
    """
    Load explicit per-client profiles instead of sampling them.

    Use this to replay a fixed heterogeneity setup — a previous run, or one
    exported from another framework. The file maps each client to its slowdown
    and bandwidth, keyed by client id or by position:

        {"Client1": {"compute_factor": 1.8, "bandwidth": 412.7}, ...}
        {"0": {"compute_factor": 1.8, "bandwidth": 412.7}, ...}

    `delay` is accepted as a synonym for `compute_factor`.

    :param client_ids: Client ids, in dispatch order.
    :param path: Path to the JSON file.
    :return: Dict mapping client_id to ClientProfile.
    """
    import json

    with open(path) as f:
        entries = json.load(f)

    profiles = {}
    for idx, cid in enumerate(client_ids):
        key = cid if cid in entries else str(idx)
        if key not in entries:
            raise KeyError(
                f"{path} has no entry for client {cid!r} (or index {idx}); "
                f"it defines {sorted(entries)[:5]}..."
            )
        entry = entries[key]
        if "compute_factor" not in entry and "delay" not in entry:
            raise KeyError(
                f"{path} entry for {key!r} needs `compute_factor` (or `delay`); "
                f"got keys {sorted(entry)}"
            )
        profiles[cid] = ClientProfile(
            compute_factor=float(entry.get("compute_factor", entry.get("delay"))),
            bandwidth=float(entry["bandwidth"]),
        )
    return profiles


def build_profiles(client_ids, sim_cfg, seed):
    """
    Sample a ClientProfile per client from the config heterogeneity block.

    Draws from a private generator rather than seeding the module-global RNG, so
    profile generation neither disturbs nor is disturbed by anything else in the
    process that uses `random` (model init, data partitioning, the drivers).
    """
    rng = random.Random(seed)
    het = sim_cfg.get("heterogeneity", {}) if sim_cfg else {}
    comp = het.get("compute", {})
    bw = het.get("bandwidth", {})
    profiles = {}
    for cid in client_ids:
        # compute_factor
        if comp.get("distribution") == "lognormal":
            pr = comp.get("params", {})
            cf = rng.lognormvariate(pr.get("mu", 0.0), pr.get("sigma", 0.5))
        else:
            cf = 1.0
        # bandwidth (Mbps)
        if bw.get("distribution") == "uniform":
            pr = bw.get("params", {})
            band = rng.uniform(pr.get("lo", 150.0), pr.get("hi", 600.0))
        else:
            band = 300.0
        profiles[cid] = ClientProfile(compute_factor=cf, bandwidth=band)
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_config", type=str, default="./vsim/config_vsim_fedasync.yaml"
    )
    parser.add_argument(
        "--client_config", type=str, default="./resources/configs/mnist/client_1.yaml"
    )
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument(
        "--seed", type=int, default=None, help="overrides simulator.seed if set"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--base_step_time",
        type=float,
        default=None,
        help="fixed per-step compute time (s); overrides config. "
        "Set for fully deterministic virtual time.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="override data partition_strategy (e.g. dirichlet_noniid)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="dirichlet alpha: Dir(alpha,...,alpha). Internally "
        "mapped to APPFL's alpha2 = alpha * num_classes.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number of classes (for alpha -> alpha2 scaling)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run virtual-time invariant checks after the simulation",
    )
    parser.add_argument(
        "--num_global_epochs",
        type=int,
        default=None,
        help="override server_configs.num_global_epochs",
    )
    parser.add_argument(
        "--max_in_flight",
        type=int,
        default=None,
        help="override simulator.max_in_flight (async: clients dispatched "
        "but not yet arrived)",
    )
    parser.add_argument(
        "--num_local_steps",
        type=int,
        default=None,
        help="override client_configs.train_configs.num_local_steps",
    )
    parser.add_argument(
        "--profiles_json",
        type=str,
        default=None,
        help="JSON file of explicit per-client {compute_factor, bandwidth}; "
        "overrides the sampled heterogeneity",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="override simulator.eval_every (0 = off)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="async",
        choices=["async", "sync_count", "sync_window"],
        help="simulation mode: async | sync_count | sync_window",
    )
    args = parser.parse_args()

    # ---- logger (console + file) ----
    log_dir = f"./vsim_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = VsimLogger(logging_id="vsim", file_dir=log_dir, file_name="vsim")

    # ---- configs ----
    server_config = OmegaConf.load(args.server_config)
    server_config.server_configs.num_clients = args.num_clients
    server_config.server_configs.device = args.device
    if args.num_global_epochs is not None:
        server_config.server_configs.num_global_epochs = args.num_global_epochs
    if args.num_local_steps is not None:
        server_config.client_configs.train_configs.num_local_steps = (
            args.num_local_steps
        )
    sim_cfg = server_config.server_configs.get("simulator", {})
    seed = args.seed if args.seed is not None else sim_cfg.get("seed", 42)
    max_in_flight = (
        args.max_in_flight
        if args.max_in_flight is not None
        else sim_cfg.get("max_in_flight", args.num_clients)
    )
    base_step_time = (
        args.base_step_time
        if args.base_step_time is not None
        else sim_cfg.get("base_step_time", None)
    )
    eval_every = (
        args.eval_every if args.eval_every is not None else sim_cfg.get("eval_every", 0)
    )
    compression_ratio = float(sim_cfg.get("compression_ratio", 1.0))

    # Keep the server-side validation dataset in sync BEFORE ServerAgent loads it
    # during __init__.
    if hasattr(server_config.server_configs, "val_data_configs"):
        vdc = server_config.server_configs.val_data_configs
        if not hasattr(vdc, "dataset_kwargs") or vdc.dataset_kwargs is None:
            vdc.dataset_kwargs = {}
        vdc.dataset_kwargs.num_clients = args.num_clients

    server_agent = ServerAgent(server_agent_config=server_config)

    client_configs = [
        OmegaConf.load(args.client_config) for _ in range(args.num_clients)
    ]
    for i in range(args.num_clients):
        client_configs[i].client_id = f"Client{i + 1}"
        client_configs[i].train_configs.device = args.device
        client_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
        client_configs[i].data_configs.dataset_kwargs.client_id = i
        client_configs[i].data_configs.dataset_kwargs.visualization = i == 0
        if args.partition is not None:
            client_configs[
                i
            ].data_configs.dataset_kwargs.partition_strategy = args.partition
        if args.alpha is not None:
            client_configs[i].data_configs.dataset_kwargs.alpha2 = (
                args.alpha * args.num_classes
            )

    client_agents = [
        ClientAgent(client_agent_config=client_configs[i])
        for i in range(args.num_clients)
    ]
    client_cfg_from_server = server_agent.get_client_configs()
    for c in client_agents:
        c.load_config(client_cfg_from_server)

    # ---- profiles + driver ----
    client_ids = [c.get_id() for c in client_agents]
    if args.profiles_json:
        profiles = build_profiles_from_json(client_ids, args.profiles_json)
    else:
        profiles = build_profiles(client_ids, sim_cfg, seed)
    logger.log_banner("Client system profiles")
    logger.log_title(["client", "slowdown", "bw_Mbps"])
    for cid, prof in profiles.items():
        logger.log_content(
            {
                "client": cid,
                "slowdown": prof.compute_factor,
                "bw_Mbps": prof.bandwidth,
            }
        )

    target_epochs = server_config.server_configs.get("num_global_epochs", 100)

    # ---- resolve mode from CLI or config ----
    mode = args.mode
    if mode == "async":
        cfg_mode = sim_cfg.get("mode", "async") if sim_cfg else "async"
        if cfg_mode.startswith("sync"):
            mode = cfg_mode

    # The simulator times raw tensor bytes, so an enabled compressor has to be
    # declared: it cannot know the achieved ratio without running the compressor.
    compressor_cfg = server_config.client_configs.get("comm_configs", {}).get(
        "compressor_configs", {}
    )
    if compressor_cfg.get("enable_compression", False) and compression_ratio == 1.0:
        logger.warning(
            "Compression is enabled in comm_configs, but simulator."
            "compression_ratio is 1.0, so transfers are modelled uncompressed and "
            "communication time is overstated. Set `compression_ratio` under "
            "`server_configs.simulator` to the fraction of the model actually sent."
        )

    common_kw = dict(
        server_agent=server_agent,
        client_agents=client_agents,
        profiles=profiles,
        logger=logger,
        seed=seed,
        base_step_time=base_step_time,
        eval_every=eval_every,
        compression_ratio=compression_ratio,
    )

    if mode == "async":
        driver = AsyncSimDriver(
            max_in_flight=max_in_flight,
            **common_kw,
        )
    else:
        sync_cfg = sim_cfg.get("sync", {}) if sim_cfg else {}
        driver = SyncSimDriver(
            participants_per_round=sync_cfg.get(
                "participants_per_round", args.num_clients
            ),
            mode="count" if mode == "sync_count" else "window",
            min_responses=sync_cfg.get("min_responses"),
            max_wait_time=sync_cfg.get("max_wait_time"),
            window_duration=sync_cfg.get("window_duration"),
            target_rounds=target_epochs,
            **common_kw,
        )

    driver.run()

    # ---- summary ----
    if mode == "async":
        accs = [
            r["val_accuracy"]
            for r in driver.history
            if isinstance(r.get("val_accuracy"), (int, float))
        ]
        if accs:
            logger.info(
                f"last per-client val_accuracy={accs[-1]:.2f} | max={max(accs):.2f}"
            )
        gevals = [
            (r["epoch"], r["global_val_accuracy"])
            for r in driver.history
            if isinstance(r.get("global_val_accuracy"), (int, float))
        ]
        if gevals:
            curve = " ".join(f"e{e}:{a:.1f}" for e, a in gevals)
            logger.info(f"GLOBAL val_accuracy curve -> {curve}")
            logger.info(
                f"GLOBAL final={gevals[-1][1]:.2f} | max={max(a for _, a in gevals):.2f}"
            )
    else:
        non_skipped = [r for r in driver.history if not r.get("skipped")]
        if non_skipped:
            avg_accepted = sum(r["accepted_count"] for r in non_skipped) / len(
                non_skipped
            )
            avg_dur = sum(r["duration"] for r in non_skipped) / len(non_skipped)
            logger.info(
                f"sync summary: {len(non_skipped)} rounds completed, "
                f"avg_accepted={avg_accepted:.1f}, avg_round_dur={avg_dur:.2f}s"
            )
        gevals = [
            (r["round"], r["global_val_accuracy"])
            for r in driver.history
            if isinstance(r.get("global_val_accuracy"), (int, float))
        ]
        if gevals:
            logger.info(
                f"GLOBAL final={gevals[-1][1]:.2f} | max={max(a for _, a in gevals):.2f}"
            )

    # ---- virtual-time invariant verification ----
    if args.verify:
        target = target_epochs
        checks = driver.verify(target)
        logger.log_banner("Virtual-time invariant checks")
        for name, ok in checks.items():
            if isinstance(ok, bool):
                line = f"  {'PASS' if ok else 'FAIL'}  {name}"
                logger.info(line) if ok else logger.error(line)
            else:
                logger.info(f"  INFO  {name} = {ok}")
        bool_checks = [v for v in checks.values() if isinstance(v, bool)]
        logger.log_banner(
            "ALL CHECKS PASSED" if all(bool_checks) else "SOME CHECKS FAILED",
            {"max_active": driver._max_active},
        )
    logger.info(f"log dir: {log_dir}")


if __name__ == "__main__":
    main()
