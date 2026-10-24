"""Executor interface for training vsim clients outside the driver's process.

The drivers keep the virtual clock and the APPFL server agent. An executor trains
clients elsewhere, on MPI workers (:class:`appfl.vsim.mpi.MPIExecutor`) or a test
double. Every duration is computed at dispatch from a fixed or calibrated per-step
time, so a driver can submit a task, queue its completion event straight away, and
block in :meth:`DeferredExecutor.result` only when the event loop reaches that
completion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskHandle:
    """
    Identifies one submitted local-training task.

    :param tid: Task id, unique per executor and increasing in submission order.
    :param cid: Client the task trains.
    :param version: Global-model version the client trains from.
    :param deadline: Virtual completion time; tasks are trained in
        ``(deadline, tid)`` order, which is the order the driver consumes them in.
    """

    tid: int
    cid: str
    version: int
    deadline: float


class DeferredExecutor(ABC):
    """
    Trains clients outside the driver's process.

    Implementations set two attributes:

    - ``client_ids``: client ids in configured order. Dispatch sampling draws from
      this list, so it must not depend on startup timing.
    - ``step_time``: calibrated seconds per local step after :meth:`setup`, or
      ``None`` when no calibration was requested.

    An executor that moves models as packed buffers can also set
    ``accepts_packed`` and override :meth:`result_packed` and :meth:`recycle`, so a
    driver that keeps the global model packed (see
    :mod:`appfl.vsim.flat_aggregation`) never copies a model on its side.
    """

    client_ids: List[str]
    step_time: Optional[float] = None
    #: Whether :meth:`setup`, :meth:`publish` and :meth:`submit` accept the global
    #: model as a :class:`~appfl.vsim.codec.PackedState`, which they then send as
    #: is. The caller never writes to that buffer again. Otherwise drivers pass a
    #: state dict.
    accepts_packed: bool = False

    @abstractmethod
    def setup(self, init_model, version: int, calibrate: bool) -> None:
        """
        Prepare the clients; called once by the driver's constructor.

        :param init_model: Initial global model.
        :param version: Version id of ``init_model``.
        :param calibrate: Measure ``step_time`` on the executor's hardware.
        """

    @abstractmethod
    def expected_steps(self, cid: str) -> int:
        """Return the number of local steps the client's next task will run."""

    @abstractmethod
    def publish(self, version: int, params) -> None:
        """
        Make global model ``version`` available to every worker at once.

        Sync rounds use this so the model is broadcast once rather than sent per
        client. Later :meth:`submit` calls for this version may pass
        ``params=None``.
        """

    @abstractmethod
    def submit(
        self, cid: str, version: int, params, deadline: float, seed: Optional[int]
    ) -> TaskHandle:
        """
        Queue local training of ``cid`` from global model ``version``.

        :param cid: Client to train.
        :param version: Version id of the global model the client trains from.
        :param params: That global model, or ``None`` if it was published.
        :param deadline: Virtual completion time of the task.
        :param seed: Task seed applied around training, or ``None``.
        :return: Handle for :meth:`result` or :meth:`discard`.
        """

    @abstractmethod
    def result(self, handle: TaskHandle) -> Tuple[Any, Dict]:
        """Block until the task has trained; return ``(local_model, meta)``."""

    def result_packed(self, handle: TaskHandle) -> Tuple[Any, Dict]:
        """
        Like :meth:`result`, but the local model may come back packed.

        An executor that receives models as packed buffers returns a
        :class:`~appfl.vsim.codec.PackedState` and hands its buffer over: the
        caller may write to it, and passes it to :meth:`recycle` once done. The
        default returns :meth:`result`.
        """
        return self.result(handle)

    def recycle(self, buffer) -> None:
        """
        Take back a model buffer the caller no longer uses, to reuse its memory.

        ``buffer`` came from :meth:`result_packed`, or was passed to
        :meth:`publish` or :meth:`submit` as a packed global model, so a send of it
        may still be in progress; the executor reuses it only after that send. The
        caller must not touch it again. The default does nothing.
        """

    @abstractmethod
    def discard(self, handle: TaskHandle) -> None:
        """Declare that the task's result will never be consumed."""

    def set_remaining(self, count: Optional[int]) -> None:
        """
        Declare that the driver will consume at most ``count`` more results.

        Results are consumed in deadline order, so the driver has consumed all
        ``count`` of them by the ``count``-th smallest pending deadline, whatever it
        submits later. A task due after that, already submitted or submitted
        later, can never be consumed: the executor may skip it, and the driver
        never asks for it. ``None`` withdraws the declaration. The default ignores
        it.

        :param count: Upper bound on the remaining :meth:`result` calls.
        """

    @abstractmethod
    def shutdown(self) -> Dict:
        """Stop the workers and return execution statistics."""
