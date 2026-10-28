"""Per-client system-heterogeneity profile for the virtual-time async FL simulator.

A single multiplicative compute slowdown factor plus a fixed per-client bandwidth.
The *measured* per-step compute time comes from APPFL's trainer metadata
(`compute_second_per_step`), so we do not time `train()` ourselves.

    compute_time = compute_second_per_step * num_local_steps * compute_factor
    comm_time    = (model_bytes + upload_bytes) * 8 / (bandwidth * 1e6)  # down + up
    duration     = compute_time + comm_time

APPFL compresses only what a client uploads; the server always sends the full
model. So `model_bytes` is the raw model, and `upload_bytes` is the update as
sent, which defaults to `model_bytes` when no compression is modelled.

Bandwidth is Mbps in its standard networking sense: 10^6 bits per second, not
2^20. Mixing the two (dividing bits by 1024**2 and then by a Mbps figure)
understates transfer time by ~4.6%.

"""

from dataclasses import dataclass
from typing import Optional

_BITS_PER_BYTE = 8
_BITS_PER_MEGABIT = 1e6  # Mbps is decimal megabits per second, not mebibits


@dataclass
class ClientProfile:
    """Per-client compute slowdown and round-trip bandwidth profile."""

    compute_factor: float = 1.0  # device slowdown multiplier (>1 is slower)
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

        :param model_bytes: Bytes actually placed on the wire.
        :return: Download duration in seconds.
        """
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * _BITS_PER_BYTE / (self.bandwidth * _BITS_PER_MEGABIT)

    def upload_time(self, model_bytes: float, **kwargs) -> float:
        """
        Compute virtual uplink communication time.

        :param model_bytes: Bytes actually placed on the wire.
        :return: Upload duration in seconds.
        """
        if self.bandwidth <= 0:
            return 0.0
        return model_bytes * _BITS_PER_BYTE / (self.bandwidth * _BITS_PER_MEGABIT)

    def comm_time(
        self, model_bytes: float, upload_bytes: Optional[float] = None, **kwargs
    ) -> float:
        """
        Compute total communication time (download + upload).

        Composed from the two directional methods rather than recomputing, so a
        profile that models asymmetric links only has to override those. An
        override of this method must accept ``upload_bytes``: otherwise it lands
        in ``**kwargs`` and the upload is timed at the download size.

        :param model_bytes: Bytes of the global model the client downloads.
        :param upload_bytes: Bytes of the update the client sends back, smaller
            than the model when compressed. Defaults to ``model_bytes``.
        :return: Round-trip communication duration in seconds.
        """
        if upload_bytes is None:
            upload_bytes = model_bytes
        return self.download_time(model_bytes, **kwargs) + self.upload_time(
            upload_bytes, **kwargs
        )

    def duration(
        self,
        compute_second_per_step: float,
        num_steps: int,
        model_bytes: float,
        upload_bytes: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Compute total virtual duration (compute + communication).

        :param compute_second_per_step: Measured per-step wall-clock time.
        :param num_steps: Number of local training steps.
        :param model_bytes: Model size in bytes, as downloaded.
        :param upload_bytes: Update size in bytes, as uploaded. Defaults to
            ``model_bytes``.
        :return: Total virtual duration in seconds.
        """
        return self.compute_time(
            compute_second_per_step, num_steps, **kwargs
        ) + self.comm_time(model_bytes, upload_bytes=upload_bytes, **kwargs)

    def next_available(self, vtime: float) -> float:
        """
        Return the earliest time at or after ``vtime`` when this client can start.

        Always ``vtime`` here — availability modelling is a later extension. That
        extension must return a *strictly later* time while the client is offline.
        A boolean "is it available?" cannot express when to retry, and re-queueing
        at the same virtual instant would spin the event loop without advancing the
        clock; returning a time makes that mistake unwritable.

        :param vtime: Virtual time at which dispatch is being attempted.
        :return: Virtual time at which the client can begin training.
        """
        return vtime
