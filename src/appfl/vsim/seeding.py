"""Per-task random seeding for the virtual-time simulator.

Local training draws from process-global RNGs: DataLoader shuffling, dropout, data
augmentation. In a serial run a client's batch order therefore depends on how many
clients happened to train before it in the same process, and on a remote worker it
would depend on placement. Seeding every task from (run seed, client, selection
count) instead makes a client's local training a function of the task alone, so the
serial engine and a distributed run produce the same trajectory.

The seed is applied inside :func:`rng_guard`, which restores the previous global RNG
state on exit. The simulator therefore neither perturbs other consumers of the
global RNGs nor is perturbed by them.
"""

import random
import sys
import hashlib
from contextlib import contextmanager

_SEED_MASK = (1 << 63) - 1


def task_seed(run_seed: int, client_id: str, count: int) -> int:
    """
    Derive the seed for one local-training task.

    BLAKE2b is used rather than ``hash()``, which is salted per process for strings,
    so every MPI rank derives the same seed for the same task.

    :param run_seed: Simulation seed.
    :param client_id: Client identifier.
    :param count: How many times this client was selected before (0-based).
    :return: Non-negative integer seed below 2**63.
    """
    key = f"{int(run_seed)}\x1f{client_id}\x1f{int(count)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") & _SEED_MASK


@contextmanager
def rng_guard(seed):
    """
    Seed the global RNGs for one task and restore their previous state afterwards.

    Python's ``random`` is always seeded. numpy and torch are seeded only when they
    are already imported, so a torch-free caller stays torch-free, and CUDA
    generators are saved and restored only when CUDA is already initialized, so the
    guard never initializes CUDA itself.

    :param seed: Seed from :func:`task_seed`, or ``None`` to leave the RNGs alone.
    """
    if seed is None:
        yield
        return

    np = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    py_state = random.getstate()
    np_state = np.random.get_state() if np is not None else None
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )

    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (1 << 32))
    if torch is not None:
        torch.manual_seed(seed)  # seeds the CUDA generators too (lazily if unused)
    try:
        yield
    finally:
        random.setstate(py_state)
        if np is not None:
            np.random.set_state(np_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
