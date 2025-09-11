import math, json
import numpy as np

def phi_staleness(s: int, mode: str, gamma: float) -> float:
    if mode == "exp": return math.exp(-gamma * max(0, s))
    return 1.0 / (1.0 + max(0, s))  # harmonic

def make_queue_sampler(cfg, client_ids):
    n = len(client_ids)
    sim = str(cfg.queue.sim_queue)
    slowdown = [float(x) for x in str(cfg.queue.slowdown).split(",")]
    assert len(slowdown) == n

    if sim == "off":
        def q_fn(cid, r): return 0.0
        return q_fn, slowdown
    if sim == "fixed":
        fixed = [float(x) for x in str(cfg.queue.queue_fixed).split(",")]
        assert len(fixed) == n
        def q_fn(cid, r): return max(0.0, fixed[client_ids.index(cid)])
        return q_fn, slowdown
    if sim == "per_round_file":
        table = json.load(open(cfg.queue.queue_file))
        def q_fn(cid, r):
            idx = client_ids.index(cid); key = str(r)
            arr = table.get(key, table.get(r, [0]*n))
            return float(arr[idx]) if arr else 0.0
        return q_fn, slowdown

    # default: lognormal
    means = [float(x) for x in str(cfg.queue.queue_means).split(",")]
    assert len(means) == n
    sigma = float(cfg.queue.queue_sigma)
    rng = np.random.RandomState(int(cfg.system.seed) + 123)
    def q_fn(cid, r):
        mu = math.log(max(1e-6, means[client_ids.index(cid)])) - 0.5 * sigma * sigma
        return float(rng.lognormal(mean=mu, sigma=sigma))
    return q_fn, slowdown
