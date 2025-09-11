import os, csv

ROUND_HEADER = [
    "round","cid","admitted","q_delay","q_hat","deadline",
    "Jk","Ek","Emin","ck","eta_k","arrival_offset","val_loss","val_acc"
]
APPLIED_HEADER = ["round_applied","cid","origin_round","staleness","phi_norm","p_k","delta_l2"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def init_logs(cfg):
    outdir = str(cfg.logging.output_dir)
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "logs"))
    csv_round = os.path.join(outdir, "logs", "round_metrics.csv")
    csv_applied = os.path.join(outdir, "logs", "applied_updates.csv")
    log_path = os.path.join(outdir, "logs", "detail.log")
    with open(csv_round, "w", newline="") as f:
        csv.writer(f).writerow(ROUND_HEADER)
    with open(csv_applied, "w", newline="") as f:
        csv.writer(f).writerow(APPLIED_HEADER)
    open(log_path, "w").close()
    return outdir, csv_round, csv_applied, log_path

def append_round_row(csv_round, row):
    with open(csv_round, "a", newline="") as f:
        csv.writer(f).writerow(row)

def append_applied_rows(csv_applied, rows):
    with open(csv_applied, "a", newline="") as f:
        wr = csv.writer(f)
        for r in rows: wr.writerow(r)

def log_line(path: str, msg: str):
    with open(path, "a") as f:
        f.write(msg + "\n")
