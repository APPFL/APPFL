"""Per-client system-heterogeneity profile for the virtual-time async FL simulator.

v1 (AFL-Lib-equivalent): a single multiplicative compute slowdown factor plus a
fixed per-client bandwidth. The *measured* per-step compute time comes from APPFL's
trainer metadata (`compute_second_per_step`), so we do not time `train()` ourselves.

    compute_time = compute_second_per_step * num_local_steps * compute_factor
    comm_time    = model_bytes * 8 / (1024**2) / bandwidth * 2   # downlink + uplink
    duration     = compute_time + comm_time

"""

from dataclasses import dataclass


@dataclass
class ClientProfile:
    """Per-client compute slowdown and round-trip bandwidth profile."""

    compute_factor: float = 1.0  # device slowdown multiplier (AFL-Lib `delay`)
    bandwidth: float = 300.0  # Mbps

    def compute_time(
        self, compute_second_per_step: float, num_steps: int, **kwargs
    ) -> float:
        """
        Compute virtual training time for this client.

        :param compute_second_per_step: Measured per-step wall-clock time (seconds).
        :param num_steps: Number of local training steps.
        :return: Virtual compute duration in seconds.
        """
        return compute_second_per_step * num_steps * self.compute_factor

    def download_time(self, model_bytes: float, **kwargs) -> float:
        """
        Compute virtual downlink communication time.

        :param model_bytes: Model size in bytes.
        :return: Download duration in seconds.
        """
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * 8 / (1024 * 1024) / self.bandwidth

    def upload_time(self, model_bytes: float, **kwargs) -> float:
        """
        Compute virtual uplink communication time.

        :param model_bytes: Model size in bytes.
        :return: Upload duration in seconds.
        """
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * 8 / (1024 * 1024) / self.bandwidth

    def comm_time(self, model_bytes: float, **kwargs) -> float:
        """
        Compute total communication time (download + upload).

        :param model_bytes: Model size in bytes.
        :return: Round-trip communication duration in seconds.
        """
        if self.bandwidth <= 0:
            return 0.0
        return (model_bytes * 8 / (1024 * 1024) / self.bandwidth) * 2

    def duration(
        self,
        compute_second_per_step: float,
        num_steps: int,
        model_bytes: float,
        **kwargs,
    ) -> float:
        """
        Compute total virtual duration (compute + communication).

        :param compute_second_per_step: Measured per-step wall-clock time.
        :param num_steps: Number of local training steps.
        :param model_bytes: Model size in bytes.
        :return: Total virtual duration in seconds.
        """
        return self.compute_time(
            compute_second_per_step, num_steps, **kwargs
        ) + self.comm_time(model_bytes, **kwargs)

    def available(self, vtime: float) -> bool:
        """Return True; availability models are added by a later extension."""
        return True
