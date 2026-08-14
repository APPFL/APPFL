"""Virtual-time FL simulation package for APPFL.

Provides event-driven simulation drivers (async, sync) with configurable
compute, communication, and availability models. Runs on a single CPU/GPU
using APPFL's public API — no framework modifications required.
"""

from .client_profile import ClientProfile
from .base_sim_driver import BaseSimDriver
from .async_sim_driver import AsyncSimDriver
from .sync_sim_driver import SyncSimDriver
from .availability_model import (
    AvailabilityModel,
    PermanentDropout,
    SessionDropout,
    CorrelatedDropout,
    TimeoutModel,
    build_availability,
)
from .comm_model import CommModel, SharedBandwidthPool, build_comm_models
from .compute_model import ComputeModel, DEVICE_PROFILES, build_compute_models

__all__ = [
    "ClientProfile",
    "BaseSimDriver",
    "AsyncSimDriver",
    "SyncSimDriver",
    "AvailabilityModel",
    "PermanentDropout",
    "SessionDropout",
    "CorrelatedDropout",
    "TimeoutModel",
    "build_availability",
    "CommModel",
    "SharedBandwidthPool",
    "build_comm_models",
    "ComputeModel",
    "DEVICE_PROFILES",
    "build_compute_models",
]
