"""Multi-node MPI runtime for the virtual-time simulator.

Rank 0 runs the event loop with an :class:`MPIExecutor`; every other rank serves
local-training tasks with :func:`run_worker`. mpi4py is imported only when one of
them is used.
"""

from .coordinator import MPIExecutor
from .worker import expected_local_steps, run_worker

__all__ = ["MPIExecutor", "expected_local_steps", "run_worker"]
