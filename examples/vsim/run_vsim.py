"""Entry point for the virtual-time asynchronous FL simulator (v1).

Async counterpart of examples/serial/run_serial.py. Reuses APPFL ServerAgent /
ClientAgent and drives them with AsyncSimDriver (virtual-time event queue).

Run from the examples/ directory (relative ./resources paths in the config):
    cd examples
    python vsim/run_vsim.py --num_clients 2 --seed 42
"""

import os
import argparse
import logging
import random
from datetime import datetime

from omegaconf import OmegaConf
from appfl.agent import ServerAgent, ClientAgent
from appfl.simulator import AsyncSimDriver, SyncSimDriver, ClientProfile
from appfl.simulator.availability_model import build_availability
from appfl.simulator.comm_model import build_comm_models
from appfl.simulator.compute_model import build_compute_models


def _counts(prop, total):
    """AFL-Lib probs_to_counts: split `total` into integer counts per proportion."""
    s = float(sum(prop))
    raw = [p / s * total for p in prop]
    fl = [int(x) for x in raw]
    rem = total - sum(fl)
    order = sorted(range(len(prop)), key=lambda i: raw[i] - fl[i], reverse=True)
    for i in range(rem):
        fl[order[i]] += 1
    return fl


def _afl_lib_profiles(client_ids, het):
    """Replicate AFL-Lib's device/comm heterogeneity exactly so settings match
    AFL-Lib: compute_factor = device_reference x SCALE_FACTOR (TX2/Nano/Pi),
    bandwidth sampled uniformly within WiFi/4G/5G tiers (utils/sys_utils.py)."""
    SCALE = 50.0
    delays = [d * SCALE for d in (1.0, 1.8125, 11.625)]  # TX2, Nano, Pi
    bands = [(150.0, 600.0), (20.0, 100.0), (50.0, 1000.0)]  # WiFi, 4G, 5G
    n = len(client_ids)
    dev_prop = list(het.get("dev_prop", [1, 1, 1]))
    comm_prop = list(het.get("comm_prop", [1, 1, 1]))
    dev_assign = [i for i, c in enumerate(_counts(dev_prop, n)) for _ in range(c)]
    comm_assign = [i for i, c in enumerate(_counts(comm_prop, n)) for _ in range(c)]
    random.shuffle(dev_assign)
    random.shuffle(comm_assign)
    profiles = {}
    for k, cid in enumerate(client_ids):
        lo, hi = bands[comm_assign[k]]
        profiles[cid] = ClientProfile(
            compute_factor=delays[dev_assign[k]], bandwidth=random.uniform(lo, hi)
        )
    return profiles


def build_profiles_from_json(client_ids, het_json):
    """Use AFL-Lib's exported per-client (delay, bandwidth) so system heterogeneity
    is IDENTICAL to AFL-Lib (delay -> compute_factor, bandwidth -> bandwidth)."""
    import json

    with open(het_json) as f:
        het = json.load(f)
    profiles = {}
    for idx, cid in enumerate(client_ids):
        h = het[str(idx)]  # AFL keys clients by integer id 0..N-1
        profiles[cid] = ClientProfile(
            compute_factor=float(h["delay"]), bandwidth=float(h["bandwidth"])
        )
    return profiles


def build_profiles(client_ids, sim_cfg, seed):
    """Sample a ClientProfile per client from the config heterogeneity block."""
    random.seed(seed)
    het = sim_cfg.get("heterogeneity", {}) if sim_cfg else {}
    if het.get("preset") == "afl_lib":
        return _afl_lib_profiles(client_ids, het)
    comp = het.get("compute", {})
    bw = het.get("bandwidth", {})
    profiles = {}
    for cid in client_ids:
        # compute_factor
        if comp.get("distribution") == "lognormal":
            pr = comp.get("params", {})
            cf = random.lognormvariate(pr.get("mu", 0.0), pr.get("sigma", 0.5))
        else:
            cf = 1.0
        # bandwidth (Mbps)
        if bw.get("distribution") == "uniform":
            pr = bw.get("params", {})
            band = random.uniform(pr.get("lo", 150.0), pr.get("hi", 600.0))
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
        "--max_concurrency",
        type=int,
        default=None,
        help="override simulator.max_concurrency (K)",
    )
    parser.add_argument(
        "--num_local_steps",
        type=int,
        default=None,
        help="override client_configs.train_configs.num_local_steps",
    )
    parser.add_argument(
        "--het_json",
        type=str,
        default=None,
        help="AFL-Lib exported per-client (delay,bandwidth) JSON to match exactly",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="override simulator.eval_every (0 = off)",
    )
    parser.add_argument(
        "--afl_dir",
        type=str,
        default=None,
        help="AFL-Lib root (to load its npz shards via afl_data.py)",
    )
    parser.add_argument(
        "--afl_dataset",
        type=str,
        default=None,
        help="AFL dataset dir name (e.g. mnist10, mnist20)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="async",
        choices=["async", "sync_count", "sync_window"],
        help="simulation mode: async | sync_count | sync_window",
    )
    parser.add_argument(
        "--timing_only",
        action="store_true",
        help="timing-only mode: skip training/aggregation, compute "
        "virtual time from profiles (requires base_step_time)",
    )
    parser.add_argument(
        "--target_epochs",
        type=int,
        default=None,
        help="target rounds for timing-only mode (overrides num_global_epochs)",
    )
    args = parser.parse_args()

    # ---- logger (console + file) ----
    log_dir = f"./vsim_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("vsim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s vsim] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, "vsim.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

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
    max_conc = (
        args.max_concurrency
        if args.max_concurrency is not None
        else sim_cfg.get("max_concurrency", args.num_clients)
    )
    base_step_time = (
        args.base_step_time
        if args.base_step_time is not None
        else sim_cfg.get("time_model", {}).get("base_step_time", None)
    )
    eval_every = (
        args.eval_every if args.eval_every is not None else sim_cfg.get("eval_every", 0)
    )

    # keep server-side global-eval dataset in sync (AFL full test set) BEFORE ServerAgent
    # loads it during __init__.
    if hasattr(server_config.server_configs, "val_data_configs"):
        vdc = server_config.server_configs.val_data_configs
        if not hasattr(vdc, "dataset_kwargs") or vdc.dataset_kwargs is None:
            vdc.dataset_kwargs = {}
        vdk = vdc.dataset_kwargs
        vdk.num_clients = args.num_clients
        if args.afl_dir is not None:
            vdk.afl_dir = args.afl_dir
        if args.afl_dataset is not None:
            vdk.dataset = args.afl_dataset

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
        if args.afl_dir is not None:
            client_configs[i].data_configs.dataset_kwargs.afl_dir = args.afl_dir
        if args.afl_dataset is not None:
            client_configs[i].data_configs.dataset_kwargs.dataset = args.afl_dataset

    client_agents = [
        ClientAgent(client_agent_config=client_configs[i])
        for i in range(args.num_clients)
    ]
    client_cfg_from_server = server_agent.get_client_configs()
    for c in client_agents:
        c.load_config(client_cfg_from_server)

    # ---- profiles + driver ----
    client_ids = [c.get_id() for c in client_agents]
    if args.het_json:
        profiles = build_profiles_from_json(client_ids, args.het_json)
    else:
        profiles = build_profiles(client_ids, sim_cfg, seed)
    for cid, prof in profiles.items():
        logger.info(
            f"profile {cid:>10}: compute_factor={prof.compute_factor:.3f}, "
            f"bandwidth={prof.bandwidth:.1f} Mbps"
        )

    # v2: comm model — sample per-client CommModel if configured
    comm_cfg = sim_cfg.get("comm_model", {}) if sim_cfg else {}
    shared_pool = None
    if comm_cfg:
        comm_models, shared_pool = build_comm_models(client_ids, comm_cfg, seed + 100)
        for cid, cm in comm_models.items():
            profiles[cid].comm = cm
            logger.info(
                f"comm   {cid:>10}: dl_bw={cm.download_bw:.1f}, ul_bw={cm.upload_bw:.1f}, "
                f"jitter={cm.jitter_sigma}, latency={cm.latency}s"
            )

    # v2: compute model — sample per-client ComputeModel if configured
    compute_cfg = sim_cfg.get("compute_model", {}) if sim_cfg else {}
    het_cfg = sim_cfg.get("heterogeneity", {}) if sim_cfg else {}
    if compute_cfg:
        compute_models = build_compute_models(
            client_ids, compute_cfg, het_cfg, seed + 300
        )
        for cid, cm in compute_models.items():
            profiles[cid].compute = cm
            logger.info(
                f"compute {cid:>10}: mode={cm.mode}, device={cm.device_type}, "
                f"factor={cm.compute_factor:.3f}"
            )

    # v2: availability / dropout
    avail_cfg = sim_cfg.get("availability", {}) if sim_cfg else {}
    availability_model, timeout_model = build_availability(
        avail_cfg, client_ids, seed + 200
    )
    if availability_model:
        logger.info(f"availability: {type(availability_model).__name__}")
    if timeout_model:
        logger.info(
            f"timeout: seconds={timeout_model.timeout}, quantile={timeout_model.quantile}"
        )

    # v2: timing-only / calibration mode
    time_model = sim_cfg.get("time_model", {}) if sim_cfg else {}
    time_mode = time_model.get("mode", None)
    calibration_epochs = None

    if args.timing_only:
        timing_only = True
    elif time_mode == "calibration":
        timing_only = False
        calibration_epochs = time_model.get("calibration_epochs", 3)
        logger.info(
            f"time_model: calibration mode ({calibration_epochs} local steps → timing-only)"
        )
    elif time_mode == "fixed":
        timing_only = True
        if base_step_time is None:
            raise ValueError("time_model.mode='fixed' requires base_step_time")
        logger.info(f"time_model: fixed mode (base_step_time={base_step_time})")
    elif time_mode == "real_measure":
        timing_only = False
        logger.info("time_model: real_measure mode (real GPU time every step)")
    else:
        timing_only = sim_cfg.get("timing_only", False)

    num_local_steps = (
        args.num_local_steps
        if args.num_local_steps is not None
        else server_config.client_configs.train_configs.get("num_local_steps", 20)
    )
    target_epochs = (
        args.target_epochs
        if args.target_epochs is not None
        else sim_cfg.get(
            "target_epochs", server_config.server_configs.get("num_global_epochs", 100)
        )
    )

    # ---- resolve mode from CLI or config ----
    mode = args.mode
    if mode == "async":
        cfg_mode = sim_cfg.get("mode", "async") if sim_cfg else "async"
        if cfg_mode.startswith("sync"):
            mode = cfg_mode

    common_kw = dict(
        server_agent=server_agent,
        client_agents=client_agents,
        profiles=profiles,
        logger=logger,
        seed=seed,
        base_step_time=base_step_time,
        eval_every=eval_every,
        availability_model=availability_model,
        timeout_model=timeout_model,
        shared_pool=shared_pool,
        timing_only=timing_only,
        num_local_steps=num_local_steps,
        calibration_epochs=calibration_epochs,
    )

    if mode == "async":
        driver = AsyncSimDriver(
            max_concurrency=max_conc,
            target_epochs=target_epochs,
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
        logger.info("VERIFY (virtual-time invariants):")
        for name, ok in checks.items():
            if isinstance(ok, bool):
                logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            else:
                logger.info(f"  [INFO] {name} = {ok}")
        bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
        logger.info(
            f"VERIFY result: {'ALL PASS' if all(bool_checks.values()) else 'SOME FAILED'} "
            f"(max_active={driver._max_active}/{driver.K})"
        )
    logger.info(f"log dir: {log_dir}")


if __name__ == "__main__":
    main()
