"""Virtual-time FL simulation package for APPFL.

Provides event-driven async and sync drivers that reconstruct target execution with
a virtual clock. Clients train serially on one CPU/GPU by default, or on many GPUs
across nodes through a :class:`DeferredExecutor` such as
:class:`appfl.vsim.mpi.MPIExecutor`.
"""

from .logger import VsimLogger
from .client_profile import ClientProfile
from .base_sim_driver import BaseSimDriver
from .async_sim_driver import AsyncSimDriver
from .sync_sim_driver import SyncSimDriver
from .executor import DeferredExecutor, TaskHandle

__all__ = [
    "VsimLogger",
    "ClientProfile",
    "BaseSimDriver",
    "AsyncSimDriver",
    "SyncSimDriver",
    "DeferredExecutor",
    "TaskHandle",
]
