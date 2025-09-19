import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import OmegaConf
from appfl.algorithm.aggregator.fedavg_aggregator import FedAvgAggregator
from .base import BaseFLAlgorithm
from utils.state_ops import sd_from_tuple, state_sub, state_add, set_server_params
from utils.delays import make_queue_sampler
from utils.io_logging import init_logs, append_round_row, append_applied_rows, log_line
from builders.appfl_builders import ensure_client_ready


class FedAvgSync(BaseFLAlgorithm):
    def run(self):
        cfg = self.cfg
        outdir, csv_round, csv_applied, log_path = init_logs(cfg)

        server = self.server
        clients = self.clients
        client_ids = self.client_ids
        n = len(client_ids)
        sizes = {}
        for cid, c in clients.items():
            try:
                sizes[cid] = int(c.get_sample_size())
            except Exception:
                sizes[cid] = 1

        total = sum(sizes.values())
        weight_mode = str(
            getattr(cfg.aggregation, "client_weight_mode", "equal")
        ).lower()

        def client_weight(cid: str) -> float:
            if weight_mode == "data_size" and total > 0:
                return sizes[cid] / float(total)
            return 1.0 / float(n)

        # ------------------ initial broadcast ------------------
        g_init = sd_from_tuple(server.get_parameters(serial_run=True))
        for c in clients.values():
            c.load_parameters(g_init)

        # ------------------ local steps per client ------------------
        if self.override_steps is not None:
            steps_vec = [int(x) for x in self.override_steps.split(",")]
            assert len(steps_vec) in (1, n), f"--steps must be single int or {n} values"
            if len(steps_vec) == 1:
                steps_vec *= n
        else:
            fedavg_cfg = getattr(cfg, "fedavg", OmegaConf.create({}))
            val = getattr(fedavg_cfg, "num_local_steps", 5)
            if isinstance(val, str) and "," in val:
                steps_vec = [int(x) for x in val.split(",")]
                assert len(steps_vec) in (1, n)
                if len(steps_vec) == 1:
                    steps_vec *= n
            else:
                steps_vec = [int(val)] * n

        steps_map = {cid: steps_vec[i] for i, cid in enumerate(client_ids)}
        eta_base = float(getattr(getattr(cfg, "optim", {}), "lr_base", 1e-3))

        # ------------------ synthetic delays ------------------
        queue_fn, slowdown = make_queue_sampler(cfg, client_ids)

        # ------------------ APPFL aggregator setup ------------------
        client_weights_mode = "sample_size" if (weight_mode == "data_size") else "equal"
        appfl_agg = FedAvgAggregator(
            model=None,
            aggregator_configs=OmegaConf.create(
                {"client_weights_mode": client_weights_mode}
            ),
            logger=None,
        )
        if client_weights_mode == "sample_size":
            for cid in client_ids:
                appfl_agg.set_client_sample_size(cid, int(sizes.get(cid, 1)))

        # ------------------ per-client worker ------------------
        def _run_one(
            cid: str,
            idx: int,
            c,
            g0_state: dict,
            steps: int,
            eta: float,
            slow_factor: float,
            q_recv: float,
        ):
            # Configure client
            c.load_parameters(g0_state)
            c.trainer.train_configs.mode = "step"
            c.trainer.train_configs.num_local_steps = int(steps)
            c.trainer.train_configs.lr = float(eta)
            opt = getattr(c.trainer, "_optimizer", None)
            if opt is not None:
                for pg in opt.param_groups:
                    pg["lr"] = float(eta)

            t0 = time.time()
            pre_wait = max(0.0, float(q_recv))
            if pre_wait > 0:
                time.sleep(pre_wait)
            t_train_start = time.time()
            dt_wait = t_train_start - t0

            # Train
            c.train()
            t_train_end = time.time()
            dt_train = max(1e-6, t_train_end - t_train_start)

            # Optional compute slowdown (heterogeneous compute)
            dt_slow = 0.0
            if slow_factor > 1.0:
                pad = dt_train * (slow_factor - 1.0)
                time.sleep(pad)
                dt_slow = pad

            arrival_ts = time.time()

            # Build delta
            w_k = c.get_parameters()
            w_k = w_k[0] if isinstance(w_k, tuple) else w_k
            delta_k = state_sub(w_k, g0_state)

            # Timings
            dt_total = arrival_ts - t0
            dt_compute = dt_train + dt_slow

            return {
                "cid": cid,
                "arrival": arrival_ts,
                "delta": delta_k,
                "q_recv": float(q_recv),
                "dt_wait": dt_wait,
                "dt_train": dt_train,
                "dt_slow": dt_slow,
                "dt_compute": dt_compute,
                "dt_total": dt_total,
            }

        # ------------------ main training rounds ------------------
        for r in range(1, int(cfg.system.num_rounds) + 1):
            t_round = time.time()
            B_r = t_round  # round start (used for arrival offset in CSV)

            # Current global
            g0 = sd_from_tuple(server.get_parameters(serial_run=True))

            # Launch clients and wait for all (strict synchronous)
            results = []
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = []
                for i, cid in enumerate(client_ids):
                    c = clients[cid]
                    ensure_client_ready(c)
                    q_recv = float(queue_fn(cid, r))  # draw delay
                    futs.append(
                        ex.submit(
                            _run_one,
                            cid,
                            i,
                            c,
                            g0,
                            steps_map[cid],
                            eta_base,
                            slowdown[i],
                            q_recv,
                        )
                    )
                for fut in as_completed(futs):
                    results.append(fut.result())

            # Canonical order
            results.sort(key=lambda x: client_ids.index(x["cid"]))

            # ------------------ aggregate via APPFL ------------------
            # Build local model states (w_k = g0 + delta_k)
            local_models = {
                res["cid"]: state_add(g0, res["delta"], 1.0) for res in results
            }
            new_global = appfl_agg.aggregate(local_models)

            # Update server
            set_server_params(server, new_global)

            # ------------------ per-client logging rows ------------------
            arrivals = {res["cid"]: res["arrival"] for res in results}
            for res in results:
                cid = res["cid"]
                append_round_row(
                    csv_round,
                    [
                        r,
                        cid,
                        1,  # round, cid, admitted
                        round(res["q_recv"], 6),
                        "",
                        "",  # q (recv), spare, spare
                        "",
                        int(steps_map[cid]),
                        "",  # spare, steps, spare
                        "",
                        round(eta_base, 8),  # spare, lr
                        round(arrivals[cid] - B_r, 6),  # arrival offset
                        "",
                        "",
                        "",  # spare
                    ],
                )

            if weight_mode == "data_size" and total > 0:
                norm_weights = [sizes[res["cid"]] / float(total) for res in results]
            else:
                norm_weights = [1.0 / float(n)] * n

            rows = []
            for w, res in zip(norm_weights, results):
                cid = res["cid"]
                rows.append(
                    [r, cid, r, 0, round(w, 8), round(client_weight(cid), 8), 0.0]
                )
            append_applied_rows(csv_applied, rows)

            # ------------------ quick validation on Client1 ------------------
            params = server.get_parameters(serial_run=True)
            clients["Client1"].load_parameters(sd_from_tuple(params))
            vloss, vacc = float("nan"), float("nan")
            tr = getattr(clients["Client1"], "trainer", None)
            if tr is not None and hasattr(tr, "_validate"):
                vloss, vacc = tr._validate()

            # ------------------ round summary ------------------
            round_wall = time.time() - t_round
            append_round_row(
                csv_round,
                [
                    r,
                    "Client1",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    round(round_wall, 6),
                    float(vloss),
                    float(vacc),
                ],
            )
            log_line(
                log_path,
                f"[FedAvg Round {r}] steps={steps_map} wall={round_wall:.2f}s val: loss={vloss:.4f} acc={vacc:.4f}",
            )

        # ------------------ save final ------------------
        import os
        import torch

        final_path = os.path.join(str(cfg.logging.output_dir), "final_global.pt")
        torch.save(sd_from_tuple(server.get_parameters(serial_run=True)), final_path)
        print("Saved:", final_path)
