import time, math, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import OmegaConf
from .base import BaseFLAlgorithm
from utils.state_ops import (
    sd_from_tuple, state_sub, state_add, state_scale, zeros_like_state,
    sd_l2norm, set_server_params
)
from utils.delays import make_queue_sampler, phi_staleness
from utils.io_logging import init_logs, append_round_row, append_applied_rows, log_line
from builders.appfl_builders import ensure_client_ready

class QueueAsyncFL(BaseFLAlgorithm):

    def run(self):
        cfg = self.cfg
        outdir, csv_round, csv_applied, log_path = init_logs(cfg)

        server = self.server
        clients = self.clients
        client_ids = self.client_ids
        n = len(client_ids)

        # weighting mode
        sizes = {}
        for cid, c in clients.items():
            try: sizes[cid] = int(c.get_sample_size())
            except Exception: sizes[cid] = 1
        total = sum(sizes.values())
        weight_mode = str(getattr(cfg.aggregation, "client_weight_mode", "equal"))
        def client_weight(cid):
            return (sizes[cid] / total) if (weight_mode == "data_size" and total>0) else (1.0/n)

        # initial broadcast
        g_init = sd_from_tuple(server.get_parameters(serial_run=True))
        for c in clients.values(): c.load_parameters(g_init)

        # warm-up throughput c_k
        warmup_steps = int(cfg.case2.warmup_steps)
        ck = {}
        for cid, c in clients.items():
            c.trainer.train_configs.mode = "step"
            c.trainer.train_configs.num_local_steps = warmup_steps
            t0 = time.time(); c.train(); dt = max(1e-6, time.time()-t0)
            ck[cid] = float(warmup_steps) / dt

        # q_hat, delays
        q_hat = {cid: float(cfg.case2.q_init) for cid in client_ids}
        queue_fn, slowdown = make_queue_sampler(cfg, client_ids)
        delay_mode = str(getattr(getattr(cfg, "algo", {}), "delay_mode", "simulate")).lower()  # simulate|sleep

        # Tsync
        if str(cfg.case2.Tsync).lower() == "auto":
            step_times = [1.0/max(1e-6, ck[cid]) for cid in client_ids]
            t_q = float(__import__("numpy").quantile(step_times, float(cfg.case2.quantile)))
            Tsync = float(cfg.case2.theta) + float(cfg.case2.q_init) + float(cfg.case2.Etarget) * t_q
            log_line(log_path, f"[auto] Tsync={Tsync:.3f}s")
        else:
            Tsync = float(cfg.case2.Tsync)

        alpha = float(cfg.case2.alpha)   # slack on Tsync
        theta = float(cfg.case2.theta)   # safety buffer inside Jk
        beta  = float(cfg.case2.beta)    # EWMA smoothing for q_hat

        phi_mode  = str(cfg.aggregation.staleness_mode)
        phi_gamma = float(cfg.aggregation.staleness_gamma)

        eta_base = float(cfg.optim.lr_base)
        lr_clip_lo = float(cfg.optim.lr_clip_lo); lr_clip_hi = float(cfg.optim.lr_clip_hi)

        broadcast_when = str(getattr(getattr(cfg, "algo", {}), "broadcast_when", "next_round")).lower()
        validate_cid   = str(getattr(getattr(cfg, "algo", {}), "validate_on_client_id", "Client1"))

        # helper to launch one client training without blocking on delay (simulate) or with sleep (sleep)
        def run_client_one_round(cid, idx, c, g0_state, Ek, eta, slow, q_comm):
            c.load_parameters(g0_state)
            c.trainer.train_configs.mode = "step"
            c.trainer.train_configs.num_local_steps = int(Ek)
            c.trainer.train_configs.lr = float(eta)
            opt = getattr(c.trainer, "_optimizer", None)
            if opt is not None:
                for pg in opt.param_groups: pg["lr"] = float(eta)
            time.sleep(max(0.0, q_comm))
            t0 = time.time()
            c.train()
            dt_compute = max(1e-6, time.time() - t0)
            if slow > 1.0:
                pad = dt_compute * (slow - 1.0)
                time.sleep(pad); dt_compute += pad
            arrival_ts = time.time()
            w_k = c.get_parameters()
            w_k = w_k[0] if isinstance(w_k, tuple) else w_k
            delta_k = state_sub(w_k, g0_state)
            return {"cid": cid, "arrival": arrival_ts, "dt_compute": dt_compute, "delta": delta_k, "q_comm": float(q_comm)}

        carry_over = []  # entries with keys: cid, delta, p, origin_round
        for r in range(1, int(cfg.system.num_rounds)+1):
            t_round = time.time()
            B_r = t_round
            g0 = sd_from_tuple(server.get_parameters(serial_run=True))

            # per-client budgets
            Jk_map, Ek_map, eta_map = {}, {}, {}
            for cid in client_ids:
                Jk = max(0.5, Tsync - q_hat[cid] - theta)
                Ek = max(1, int(math.floor(ck[cid] * Jk)))
                Jk_map[cid] = Jk; Ek_map[cid] = Ek
            Emin = max(1, min(Ek_map.values()))
            for cid in client_ids:
                eta = eta_base * (Emin / float(Ek_map[cid]))
                eta_map[cid] = max(eta_base*lr_clip_lo, min(eta, eta_base*lr_clip_hi))

            # sample comm delays once
            q_comm_map = {cid: float(queue_fn(cid, r)) for cid in client_ids}

            # run clients in parallel
            results = []
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futs = []
                for i, cid in enumerate(client_ids):
                    c = clients[cid]; ensure_client_ready(c)
                    futs.append(ex.submit(
                        run_client_one_round,
                        cid, i, c, g0,
                        Ek_map[cid],
                        eta_map[cid],
                        slowdown[i],
                        q_comm_map[cid],
                    ))
                for fut in as_completed(futs):
                    results.append(fut.result())

            # admission by relaxed horizon and q_hat update
            admitted_now, late_this_round = [], []
            arrivals = {}
            for res in results:
                cid = res["cid"]
                arrivals[cid] = res["arrival"]
                q_delay = res["q_comm"]
                # EWMA
                q_hat[cid] = (1.0 - beta) * q_hat[cid] + beta * q_delay
                # relaxed horizon per client
                deadline = q_hat[cid] + alpha * Tsync
                admitted = (q_delay <= deadline)

                item = {"cid": cid, "delta": res["delta"], "p": client_weight(cid), "origin_round": r, "q_delay": q_delay,
                        "Jk": Jk_map[cid], "Ek": Ek_map[cid], "eta": eta_map[cid], "arrival_offset": arrivals[cid]-B_r}
                if admitted:
                    admitted_now.append(item)
                else:
                    late_this_round.append(item)

                # per-client CSV row
                append_round_row(csv_round, [
                    r, cid, int(admitted),
                    round(q_delay, 6), round(q_hat[cid], 6), round(deadline, 6),
                    round(Jk_map[cid], 6), int(Ek_map[cid]), int(Emin),
                    round(ck[cid], 6), round(eta_map[cid], 8),
                    round(item["arrival_offset"], 6), "", "", ""
                ])

            # aggregate admitted + carry_over (s=0 for admitted, s=1 for carry-over)
            agg_items = []
            for it in carry_over:
                w = it["p"] * phi_staleness(1, phi_mode, phi_gamma)
                agg_items.append({"cid": it["cid"], "delta": it["delta"], "w": w, "origin_round": it["origin_round"], "staleness": 1})
            for it in admitted_now:
                w = it["p"] * phi_staleness(0, phi_mode, phi_gamma)
                agg_items.append({"cid": it["cid"], "delta": it["delta"], "w": w, "origin_round": it["origin_round"], "staleness": 0})

            if len(agg_items) > 0:
                W = sum(x["w"] for x in agg_items) or 1.0
                accum = zeros_like_state(g0)
                for x in agg_items:
                    accum = state_add(accum, state_scale(x["delta"], x["w"]/W), alpha=1.0)
                new_global = state_add(g0, accum, alpha=1.0)
                set_server_params(server, new_global)

                # applied updates log
                rows = []
                for x in agg_items:
                    rows.append([r, x["cid"], x["origin_round"], x["staleness"],
                                 round(x["w"]/W, 8),
                                 round(client_weight(x["cid"]), 8),
                                 round(sd_l2norm(x["delta"]), 6)])
                append_applied_rows(csv_applied, rows)

                # optional immediate broadcast to admitted clients only
                if broadcast_when == "immediate":
                    for it in admitted_now:
                        clients[it["cid"]].load_parameters(new_global)
            else:
                # no-op aggregation
                set_server_params(server, g0)

            # validation on chosen client
            vloss, vacc = float("nan"), float("nan")
            if validate_cid in clients:
                params = server.get_parameters(serial_run=True)
                clients[validate_cid].load_parameters(sd_from_tuple(params))
                tr = getattr(clients[validate_cid], "trainer", None)
                if tr is not None and hasattr(tr, "_validate"):
                    vloss, vacc = tr._validate()
            append_round_row(csv_round, [r, validate_cid, "", "", "", "", "", "", "", "", "", "", float(vloss), float(vacc)])

            # logs
            late_str = ", ".join([f"{x['cid']}[from={x['origin_round']}]"
                                  for x in carry_over]) if carry_over else "[]"
            log_line(log_path, f"[Round {r}] Tsync={Tsync:.2f}, alpha={alpha}, theta={theta}, carried_in={len(carry_over)}; late_out={len(late_this_round)}")
            log_line(log_path, f"  A_on_time(r={r}) = {[it['cid'] for it in admitted_now]}")
            log_line(log_path, f"  A_late(r={r})   = {late_str}")
            for cid in client_ids:
                log_line(log_path, f"  {cid}: q_hat={q_hat[cid]:.3f}, Jk={Jk_map[cid]:.3f}, Ek={Ek_map[cid]}, ck={ck[cid]:.2f}, eta={eta_map[cid]:.5f}")
            log_line(log_path, f"  val[{validate_cid}]: loss={vloss:.4f} acc={vacc:.4f}")

            # carry over late
            carry_over = [{"cid": it["cid"], "delta": it["delta"], "p": it["p"], "origin_round": it["origin_round"]}
                          for it in late_this_round]

        # save final global
        import os, torch
        final_path = os.path.join(str(cfg.logging.output_dir), "final_global.pt")
        torch.save(sd_from_tuple(server.get_parameters(serial_run=True)), final_path)
        print("Saved:", final_path)
