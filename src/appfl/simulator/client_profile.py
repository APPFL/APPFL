"""Per-client system-heterogeneity profile for the virtual-time async FL simulator.

v1 (AFL-Lib-equivalent): a single multiplicative compute slowdown factor plus a
fixed per-client bandwidth. The *measured* per-step compute time comes from APPFL's
trainer metadata (`compute_second_per_step`), so we do not time `train()` ourselves.

    compute_time = compute_second_per_step * num_local_steps * compute_factor
    comm_time    = model_bytes * 8 / (1024**2) / bandwidth * 2   # downlink + uplink
    duration     = compute_time + comm_time

v2 additions:
  - Optional CommModel (asymmetric BW, jitter, congestion, compression, TCP overhead).
  - Optional ComputeModel (device profiles, FLOPs-based, multiple modes).
  - `available()` remains always-True; dropout is via driver-level AvailabilityModel.
"""

from dataclasses import dataclass


@dataclass
class ClientProfile:
    compute_factor: float = 1.0   # device slowdown multiplier (AFL-Lib `delay`)
    bandwidth: float = 300.0      # Mbps, v1 legacy (used when comm is None)
    comm: object = None           # Optional CommModel (v2)
    compute: object = None        # Optional ComputeModel (v2)

    def compute_time(self, compute_second_per_step: float, num_steps: int, **kwargs) -> float:
        if self.compute is not None:
            return self.compute.compute_time(compute_second_per_step, num_steps, **kwargs)
        return compute_second_per_step * num_steps * self.compute_factor

    def download_time(self, model_bytes: float, **kwargs) -> float:
        if self.comm is not None:
            return self.comm.download_time(model_bytes, **kwargs)
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * 8 / (1024 * 1024) / self.bandwidth

    def upload_time(self, model_bytes: float, **kwargs) -> float:
        if self.comm is not None:
            return self.comm.upload_time(model_bytes, **kwargs)
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * 8 / (1024 * 1024) / self.bandwidth

    def comm_time(self, model_bytes: float, **kwargs) -> float:
        if self.comm is not None:
            return self.comm.comm_time(model_bytes, **kwargs)
        if self.bandwidth <= 0:
            return 0.0
        return (model_bytes * 8 / (1024 * 1024) / self.bandwidth) * 2

    def duration(self, compute_second_per_step: float, num_steps: int,
                 model_bytes: float, **kwargs) -> float:
        return self.compute_time(compute_second_per_step, num_steps, **kwargs) + self.comm_time(model_bytes, **kwargs)

    def available(self, vtime: float) -> bool:
        return True
