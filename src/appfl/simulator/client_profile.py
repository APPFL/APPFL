"""Per-client system-heterogeneity profile for the virtual-time async FL simulator.

v1 (AFL-Lib-equivalent): a single multiplicative compute slowdown factor plus a
fixed per-client bandwidth. The *measured* per-step compute time comes from APPFL's
trainer metadata (`compute_second_per_step`), so we do not time `train()` ourselves.

    compute_time = compute_second_per_step * num_local_steps * compute_factor
    comm_time    = model_bytes * 8 / (1024**2) / bandwidth * 2   # downlink + uplink
    duration     = compute_time + comm_time

This mirrors AFL-Lib's `@time_record` (execution_time * delay + comm) while reusing
APPFL's native measurement. Heterogeneity knobs (data/model-dependent compute,
time-varying bandwidth, availability) are intentionally deferred to v2 (see note 03).
"""

from dataclasses import dataclass


@dataclass
class ClientProfile:
    compute_factor: float = 1.0  # device slowdown multiplier (AFL-Lib `delay`)
    bandwidth: float = 300.0     # Mbps, fixed in v1

    def compute_time(self, compute_second_per_step: float, num_steps: int) -> float:
        """Simulated local-compute time for this (slower/faster) device."""
        return compute_second_per_step * num_steps * self.compute_factor

    def comm_time(self, model_bytes: float) -> float:
        """Round-trip (downlink + uplink) communication time in seconds."""
        if self.bandwidth <= 0:
            return 0.0
        return (model_bytes * 8 / (1024 * 1024) / self.bandwidth) * 2

    def duration(self, compute_second_per_step: float, num_steps: int, model_bytes: float) -> float:
        """Total virtual time from dispatch to completion = compute + comm."""
        return self.compute_time(compute_second_per_step, num_steps) + self.comm_time(model_bytes)

    def available(self, vtime: float) -> bool:
        """v1: always available. (Real availability/session/dropout is v2 — note 03 D.)"""
        return True
