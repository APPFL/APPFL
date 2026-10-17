"""Unit tests for per-task seeding in the virtual-time simulator (appfl.vsim.seeding)."""

import os
import random
import subprocess
import sys

from appfl.vsim.seeding import rng_guard, task_seed


def test_task_seed_is_deterministic_and_distinct():
    seeds = {task_seed(42, cid, k) for cid in ("Client1", "Client2") for k in range(3)}
    assert len(seeds) == 6
    assert task_seed(42, "Client1", 0) == task_seed(42, "Client1", 0)
    assert task_seed(42, "Client1", 0) != task_seed(43, "Client1", 0)
    assert all(0 <= s < 2**63 for s in seeds)


def test_task_seed_does_not_depend_on_hash_randomization():
    """Every MPI rank must derive the same seed, whatever its PYTHONHASHSEED."""
    code = "from appfl.vsim.seeding import task_seed; print(task_seed(7, 'Client3', 5))"
    outputs = set()
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert outputs == {str(task_seed(7, "Client3", 5))}


def test_rng_guard_restores_the_global_random_stream():
    random.seed(1234)
    expected = [random.random() for _ in range(3)]
    random.seed(1234)
    with rng_guard(99):
        random.random()
    assert [random.random() for _ in range(3)] == expected


def test_rng_guard_makes_draws_a_function_of_the_seed():
    def draws(seed):
        with rng_guard(seed):
            return [random.random() for _ in range(3)]

    random.seed(0)
    a = draws(5)
    random.random()  # whatever happened before must not matter
    assert draws(5) == a
    assert draws(6) != a


def test_rng_guard_none_is_a_no_op():
    random.seed(3)
    expected = random.random()
    random.seed(3)
    with rng_guard(None):
        pass
    assert random.random() == expected


def test_rng_guard_seeds_and_restores_numpy_and_torch_when_imported():
    import numpy as np
    import torch

    np.random.seed(1)
    torch.manual_seed(1)
    np_next, torch_next = np.random.rand(), torch.rand(1)

    np.random.seed(1)
    torch.manual_seed(1)
    with rng_guard(11):
        inside = (np.random.rand(), torch.rand(1))
    assert np.random.rand() == np_next
    assert torch.equal(torch.rand(1), torch_next)

    with rng_guard(11):
        assert np.random.rand() == inside[0]
        assert torch.equal(torch.rand(1), inside[1])


def test_rng_guard_restores_state_when_the_task_raises():
    random.seed(8)
    expected = random.random()
    random.seed(8)
    try:
        with rng_guard(1):
            raise ValueError("boom")
    except ValueError:
        pass
    assert random.random() == expected
