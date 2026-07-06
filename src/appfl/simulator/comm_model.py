"""Communication model for the virtual-time async FL simulator (v2).

Replaces v1's deterministic ``(bytes * 8 / BW) * 2`` with:
  - Asymmetric upload / download bandwidth
  - Lognormal jitter (stochastic per-round comm time)
  - Shared bandwidth pool (congestion when K clients communicate)
  - Compression ratio (simulated, not actual compress/decompress)
  - TCP protocol overhead (SimGrid LV08 model)

Backward compatible: default params reproduce v1 behavior exactly.
"""

from dataclasses import dataclass
import random as _random


class SharedBandwidthPool:
    """Fair-share bandwidth pool shared across concurrently communicating clients."""

    def __init__(self, total_bandwidth=1000.0, mode="fair_share"):
        self.total_bw = total_bandwidth
        self.mode = mode

    def effective_bandwidth(self, client_bw, num_concurrent):
        """
        Compute effective bandwidth under congestion.

        :param client_bw: Individual client bandwidth (Mbps).
        :param num_concurrent: Number of concurrently communicating clients.
        :return: Effective bandwidth (Mbps).
        """
        if self.mode == "none" or num_concurrent <= 1:
            return client_bw
        fair_share = self.total_bw / num_concurrent
        return min(client_bw, fair_share)


@dataclass
class CommModel:
    """Per-client communication model."""

    download_bw: float = 300.0  # Mbps
    upload_bw: float = 300.0  # Mbps (v1: symmetric, same as download)
    jitter_sigma: float = 0.0  # 0 = deterministic (v1 compat)
    compression_ratio: float = 1.0  # 1.0 = no compression
    latency: float = 0.0  # base RTT in seconds (for TCP overhead)

    def _effective_bw(self, bw, num_concurrent, shared_pool):
        if shared_pool and num_concurrent > 1:
            return shared_pool.effective_bandwidth(bw, num_concurrent)
        return bw

    def _direction_time(
        self, model_bytes, bw, num_concurrent=1, shared_pool=None, rng=None
    ):
        """Compute one-direction transfer time with jitter, congestion, and TCP overhead."""
        effective_bytes = model_bytes / max(self.compression_ratio, 0.01)
        bits_mb = effective_bytes * 8 / (1024 * 1024)
        eff_bw = self._effective_bw(bw, num_concurrent, shared_pool)
        overhead = self.latency * 13.01 if self.latency > 0 else 0.0
        base_time = overhead + bits_mb / max(eff_bw, 0.001)
        if self.jitter_sigma > 0 and rng:
            base_time *= rng.lognormvariate(0, self.jitter_sigma)
        return max(base_time, 0.0)

    def download_time(self, model_bytes, num_concurrent=1, shared_pool=None, rng=None):
        """
        Compute virtual download time.

        :param model_bytes: Model size in bytes.
        :param num_concurrent: Number of concurrently communicating clients.
        :param shared_pool: SharedBandwidthPool (optional).
        :param rng: Random instance for jitter (optional).
        :return: Download duration in seconds.
        """
        return self._direction_time(
            model_bytes, self.download_bw, num_concurrent, shared_pool, rng
        )

    def upload_time(self, model_bytes, num_concurrent=1, shared_pool=None, rng=None):
        """
        Compute virtual upload time.

        :param model_bytes: Model size in bytes.
        :param num_concurrent: Number of concurrently communicating clients.
        :param shared_pool: SharedBandwidthPool (optional).
        :param rng: Random instance for jitter (optional).
        :return: Upload duration in seconds.
        """
        return self._direction_time(
            model_bytes, self.upload_bw, num_concurrent, shared_pool, rng
        )

    def comm_time(self, model_bytes, num_concurrent=1, shared_pool=None, rng=None):
        """
        Compute total round-trip communication time (download + upload).

        :param model_bytes: Model size in bytes.
        :param num_concurrent: Number of concurrently communicating clients.
        :param shared_pool: SharedBandwidthPool (optional).
        :param rng: Random instance for jitter (optional).
        :return: Total communication duration in seconds.
        """
        return self.download_time(
            model_bytes, num_concurrent, shared_pool, rng
        ) + self.upload_time(model_bytes, num_concurrent, shared_pool, rng)


def _sample_dist(cfg, rng, default=300.0):
    if not cfg:
        return default
    dist = cfg.get("distribution", "fixed")
    params = cfg.get("params", {})
    if dist == "uniform":
        return rng.uniform(params.get("lo", default), params.get("hi", default))
    elif dist == "lognormal":
        return rng.lognormvariate(params.get("mu", 0), params.get("sigma", 0.5))
    elif dist == "fixed":
        return params.get("value", default)
    return default


def build_comm_models(client_ids, comm_cfg, seed):
    """Sample a CommModel per client from config.  Returns ({cid: CommModel}, SharedBandwidthPool|None)."""
    if not comm_cfg:
        return {}, None

    rng = _random.Random(seed)
    dl_cfg = comm_cfg.get("download_bw", {})
    ul_cfg = comm_cfg.get("upload_bw", {})
    jitter = comm_cfg.get("jitter_sigma", 0.0)
    compression = comm_cfg.get("compression_ratio", 1.0)
    latency = comm_cfg.get("latency", 0.0)

    models = {}
    for cid in client_ids:
        dl = _sample_dist(dl_cfg, rng, 300.0)
        ul = _sample_dist(ul_cfg, rng, dl) if ul_cfg else dl
        models[cid] = CommModel(
            download_bw=dl,
            upload_bw=ul,
            jitter_sigma=jitter,
            compression_ratio=compression,
            latency=latency,
        )

    pool_cfg = comm_cfg.get("shared_pool", {})
    pool = None
    if pool_cfg:
        pool = SharedBandwidthPool(
            total_bandwidth=pool_cfg.get("total_bandwidth", 1000.0),
            mode=pool_cfg.get("mode", "fair_share"),
        )

    return models, pool
