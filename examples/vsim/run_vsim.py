"""Entry point for the virtual-time FL simulator (async and sync).

Async counterpart of examples/serial/run_serial.py. Reuses APPFL ServerAgent /
ClientAgent and drives them with AsyncSimDriver or SyncSimDriver (virtual-time
event queue).

Run from the examples/ directory (relative ./resources paths in the config):
    cd examples
    python vsim/run_vsim.py --num_clients 2 --seed 42

Multi-GPU / multi-node: launch under MPI with ``--executor mpi``. Rank 0 runs the
event loop and the server; every other rank hosts a share of the clients on one
device (one rank per GPU):
    mpiexec -n 5 python vsim/run_vsim.py --executor mpi --device cuda \
        --num_clients 100 --verify
"""

import argparse
import gc
import json
import math
import os
import random
import traceback
from datetime import datetime

from omegaconf import OmegaConf
from appfl.agent import ServerAgent, ClientAgent
from appfl.algorithm.scheduler import AsyncScheduler
from appfl.vsim import AsyncSimDriver, SyncSimDriver, ClientProfile, VsimLogger


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


def estimate_durations(profiles, train_configs, step_time, model_bytes):
    """
    Estimate every client's task duration before any client is built.

    The MPI executor uses the estimates only to place clients on workers, so they
    must rank clients and weigh compute against communication, but need not equal
    the simulated durations. A calibrated step time is unknown until the workers
    start, so without `base_step_time` communication is left out.

    :return: Dict mapping client_id to seconds, or None in epoch mode, where the
        step count depends on each client's data.
    """
    if train_configs.get("mode") != "step":
        return None
    steps = int(train_configs.num_local_steps)
    if step_time is None:
        return {cid: p.compute_time(1.0, steps) for cid, p in profiles.items()}
    return {
        cid: p.duration(float(step_time), steps, model_bytes)
        for cid, p in profiles.items()
    }


def parse_args():
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
    parser.add_argument(
        "--executor",
        type=str,
        default=None,
        choices=["inprocess", "mpi"],
        help="where clients train: inprocess (serial, default) or mpi (rank 0 runs "
        "the event loop, every other rank hosts clients; launch with mpiexec); "
        "overrides simulator.executor",
    )
    parser.add_argument(
        "--save_history",
        type=str,
        default=None,
        help="write driver.history to this JSON file (e.g. to compare runs)",
    )
    return parser.parse_args()


def make_client_config(args, index, device, file_logging=True):
    """
    Build the configuration of client ``index`` (0-based) from --client_config.

    :param file_logging: Keep the client's log file. MPI workers turn it off:
        every ClientAgent opens four file handlers, so a worker hosting hundreds
        of clients would run out of file descriptors.
    """
    config = OmegaConf.load(args.client_config)
    config.client_id = f"Client{index + 1}"
    config.train_configs.device = device
    config.data_configs.dataset_kwargs.num_clients = args.num_clients
    config.data_configs.dataset_kwargs.client_id = index
    config.data_configs.dataset_kwargs.visualization = index == 0
    if args.partition is not None:
        config.data_configs.dataset_kwargs.partition_strategy = args.partition
    if args.alpha is not None:
        config.data_configs.dataset_kwargs.alpha2 = args.alpha * args.num_classes
    if not file_logging:
        config.train_configs.logging_output_filename = ""
    return config


def make_client(args, index, server_client_configs, device, file_logging=True):
    """Create client ``index`` and load the configuration shared by the server."""
    client = ClientAgent(
        client_agent_config=make_client_config(args, index, device, file_logging)
    )
    client.load_config(server_client_configs)
    return client


def serve_worker(args, comm):
    """Run this rank as an MPI worker until the coordinator shuts it down."""
    from appfl.vsim.mpi import run_worker

    index = {f"Client{i + 1}": i for i in range(args.num_clients)}

    def make_worker_client(cid, context, device):
        return make_client(
            args, index[cid], OmegaConf.create(context), device, file_logging=False
        )

    run_worker(comm, make_worker_client, device=args.device)


def main():
    args = parse_args()

    server_config = OmegaConf.load(args.server_config)
    sim_cfg = server_config.server_configs.get("simulator", {})
    executor_kind = args.executor or (sim_cfg.get("executor") if sim_cfg else None)
    executor_kind = executor_kind or "inprocess"

    # ---- logger (console + file); with MPI, rank 0 picks the run directory ----
    comm = None
    log_dir = f"./vsim_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if executor_kind == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        log_dir = comm.bcast(log_dir, root=0)
        if comm.Get_rank() != 0:
            serve_worker(args, comm)
            return
    logger = VsimLogger(logging_id="vsim", file_dir=log_dir, file_name="vsim")
    try:
        run_coordinator(args, server_config, sim_cfg, comm, log_dir, logger)
    except BaseException:
        if comm is None:
            raise
        # Workers would otherwise wait for this rank forever.
        logger.error(f"simulation failed:\n{traceback.format_exc()}")
        comm.Abort(1)


def run_coordinator(args, server_config, sim_cfg, comm, log_dir, logger):
    """Build the server, clients (or MPI executor) and driver; run and report."""
    # ---- configs ----
    server_config.server_configs.num_clients = args.num_clients
    server_config.server_configs.device = args.device
    if args.num_global_epochs is not None:
        server_config.server_configs.num_global_epochs = args.num_global_epochs
    if args.num_local_steps is not None:
        server_config.client_configs.train_configs.num_local_steps = (
            args.num_local_steps
        )
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
    client_cfg_from_server = server_agent.get_client_configs()
    client_ids = [f"Client{i + 1}" for i in range(args.num_clients)]
    target_epochs = server_config.server_configs.get("num_global_epochs", 100)

    # ---- resolve mode from CLI or config ----
    mode = args.mode
    if mode == "async":
        cfg_mode = sim_cfg.get("mode", "async") if sim_cfg else "async"
        if cfg_mode.startswith("sync"):
            mode = cfg_mode

    # ---- profiles (before the executor, which places clients by them) ----
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

    executor, client_agents = None, None
    if comm is not None:
        from appfl.vsim.mpi import MPIExecutor

        if args.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
            logger.warning(
                "OMP_NUM_THREADS is not set, so every CPU worker uses one torch "
                "thread. Set it to choose another count, and use the same value "
                "for an in-process run whose models should match bit for bit."
            )
        model = server_agent.model
        model_bytes = 0.0
        if model is not None:
            model_bytes = compression_ratio * sum(
                t.numel() * t.element_size() for t in model.state_dict().values()
            )
        mpi_cfg = sim_cfg.get("mpi", {}) if sim_cfg else {}
        executor = MPIExecutor(
            comm,
            client_ids,
            # Plain containers pickle cleanly; the *_source entries carry the
            # model/loss/metric code, so workers need no access to those files.
            context=OmegaConf.to_container(client_cfg_from_server, resolve=True),
            log_dir=log_dir,
            calibration_runs=mpi_cfg.get("calibration_runs", 2),
            prefetch_mb=mpi_cfg.get("prefetch_mb", 1024),
            stall_timeout=mpi_cfg.get("stall_timeout", 1800),
            logger=logger,
            # Balance the expected work per worker rather than the client count.
            durations=estimate_durations(
                profiles,
                client_cfg_from_server.train_configs,
                base_step_time,
                model_bytes,
            ),
            mode=mode,
            max_in_flight=max_in_flight,
        )
    else:
        client_agents = [
            make_client(args, i, client_cfg_from_server, args.device)
            for i in range(args.num_clients)
        ]

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
        executor=executor,
        task_seeding=bool(sim_cfg.get("task_seeding", True)) if sim_cfg else True,
    )

    if mode == "async":
        driver = AsyncSimDriver(
            max_in_flight=max_in_flight,
            # AsyncScheduler counts one epoch per arrival, so the run ends after
            # num_global_epochs arrivals and an executor can skip tasks past them.
            target_epochs=(
                target_epochs
                if isinstance(server_agent.scheduler, AsyncScheduler)
                else None
            ),
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

    if executor is not None and mode == "async":
        per_worker = math.ceil(max_in_flight / executor.num_workers)
        logger.info(
            f"in-flight model state: about {per_worker} model(s), "
            f"{per_worker * driver._raw_model_bytes / 2**20:.1f} MiB, per worker"
        )
        if max_in_flight < 4 * executor.num_workers:
            logger.warning(
                f"max_in_flight={max_in_flight} is below 4x the {executor.num_workers} "
                f"workers; with clients pinned to their home workers some workers "
                f"will idle"
            )

    # APPFL calls gc.collect() on every update; freezing the long-lived objects
    # built so far keeps each collection cheap (the MPI executor does this too).
    gc.collect()
    gc.freeze()
    driver.run()
    if executor is not None:
        stats = executor.shutdown()
        logger.log_banner(
            "Execution summary",
            {
                "workers": stats["workers"],
                "setup": f"{stats['setup_seconds']:.1f}s",
                "wall": f"{stats['wall_seconds']:.1f}s",
                "utilization": f"{100 * stats['worker_utilization']:.0f}%",
                "stall": f"{stats['stall_seconds']:.1f}s",
                "trained": stats["trained"],
                "used": stats["results"],
                "dropped": stats["dropped"],
                "model_sends": stats["model_sends"],
                "sends_saved": stats["model_sends_saved"],
                "broadcasts": stats["broadcasts"],
                "MB_out": f"{stats['bytes_to_workers'] / 1e6:.1f}",
                "MB_in": f"{stats['bytes_from_workers'] / 1e6:.1f}",
            },
        )

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
    if args.save_history:
        with open(args.save_history, "w") as f:
            json.dump(driver.history, f, indent=1)
        logger.info(f"history written to {args.save_history}")
    logger.info(f"log dir: {log_dir}")


if __name__ == "__main__":
    main()
