"""Tests for ``appfl.misc.data.cache_per_process``.

The example dataset loaders are executed afresh for every client, so they keep
their test set and training-set partition in this process-wide cache instead of
rebuilding them per client. A wrong hit would hand a client another partition's
data, and a hit that skipped the build's RNG side effect would change every random
number drawn after client construction.
"""

import gc
import weakref

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from appfl.misc.data import (
    cache_per_process,
    class_noniid_partition,
    dirichlet_noniid_partition,
    iid_partition,
)
from appfl.misc.utils import run_function_from_file


@pytest.fixture(autouse=True)
def restore_numpy_rng():
    """These tests reseed NumPy's global RNG; leave it as other tests expect."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


class RecordingBuild:
    """Build function that records its keyword arguments and returns a new list."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [len(self.calls)]


class _Labeled(Dataset):
    """Ten classes of 2x2 images; every sample has a distinct input."""

    def __init__(self, per_class: int = 12):
        labels = [c for c in range(10) for _ in range(per_class)]
        self._x = torch.arange(len(labels) * 4, dtype=torch.float32).reshape(
            len(labels), 1, 2, 2
        )
        self._y = labels

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]


def test_same_arguments_reuse_the_first_value():
    build = RecordingBuild()
    first = cache_per_process(
        "same-args", build, num_clients=4, Cmin={1: 10, "none": 3}
    )
    # Clients get their kwargs from OmegaConf, in whatever order the YAML has.
    again = cache_per_process(
        "same-args",
        build,
        num_clients=4,
        Cmin=OmegaConf.create({"none": 3, 1: 10}),
    )
    assert again is first
    assert len(build.calls) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"num_clients": 5},
        {"partition_strategy": "iid"},
        {"seed": 1},
        {"Cmin": {1: 9, "none": 3}},
    ],
)
def test_changed_arguments_rebuild(changed):
    base = dict(
        num_clients=4,
        partition_strategy="class_noniid",
        seed=42,
        Cmin={1: 10, "none": 3},
    )
    build = RecordingBuild()
    first = cache_per_process(("changed-args", repr(changed)), build, **base)
    second = cache_per_process(
        ("changed-args", repr(changed)), build, **{**base, **changed}
    )
    assert second == [2]
    assert first == [1]
    assert build.calls[-1] == {**base, **changed}


def test_hit_leaves_numpy_rng_where_the_build_left_it():
    def build(seed):
        np.random.seed(seed)
        return np.random.rand(3)

    cache_per_process("rng-replay", build, seed=7)
    np.random.seed(0)  # something draws from the RNG between two clients
    cache_per_process("rng-replay", build, seed=7)
    after_hit = np.random.rand()

    np.random.seed(7)
    np.random.rand(3)
    assert after_hit == np.random.rand()


def test_hit_leaves_numpy_rng_alone_if_the_build_does_not_use_it():
    build = RecordingBuild()
    cache_per_process("rng-free", build, num_clients=4)
    np.random.seed(0)
    cache_per_process("rng-free", build, num_clients=4)
    after_hit = np.random.rand()

    np.random.seed(0)
    assert after_hit == np.random.rand()


def test_figure_arguments_do_not_cause_a_rebuild():
    # The runners set visualization only for client 0.
    build = RecordingBuild()
    kwargs = dict(num_clients=4, output_dirname="out", output_filename="dist.pdf")
    first = cache_per_process("figure-args", build, visualization=True, **kwargs)
    for _ in range(3):
        assert (
            cache_per_process("figure-args", build, visualization=False, **kwargs)
            is first
        )
    assert len(build.calls) == 1
    assert build.calls[0]["visualization"] is True


def test_hit_draws_a_requested_figure_once():
    build = RecordingBuild()
    kwargs = dict(num_clients=4, output_dirname="out")
    first = cache_per_process("figure-draw", build, visualization=False, **kwargs)

    # A figure nobody drew yet: the build runs again to draw it.
    again = cache_per_process(
        "figure-draw", build, visualization=True, output_filename="a.pdf", **kwargs
    )
    assert again is first
    assert len(build.calls) == 2
    assert build.calls[1]["visualization"] is True
    assert build.calls[1]["output_filename"] == "a.pdf"

    # The same figure again is not redrawn; a different file is.
    cache_per_process(
        "figure-draw", build, visualization=True, output_filename="a.pdf", **kwargs
    )
    assert len(build.calls) == 2
    cache_per_process(
        "figure-draw", build, visualization=True, output_filename="b.pdf", **kwargs
    )
    assert len(build.calls) == 3


def test_changing_arguments_releases_the_previous_value():
    class Value:
        pass

    first = cache_per_process("release", lambda **kwargs: Value(), num_clients=4)
    first_ref = weakref.ref(first)
    del first
    cache_per_process("release", lambda **kwargs: Value(), num_clients=5)
    gc.collect()
    assert first_ref() is None


def test_value_survives_reexecution_of_the_loader_file(tmp_path):
    # ClientAgent runs the loader through run_function_from_file, which executes
    # the file afresh for every client, so `build` is a new function each time.
    loader = tmp_path / "loader.py"
    loader.write_text(
        "from appfl.misc.data import cache_per_process\n"
        "\n"
        "\n"
        "def build(num_clients):\n"
        "    return [num_clients]\n"
        "\n"
        "\n"
        "def get_data(num_clients):\n"
        "    return cache_per_process('reexecuted', build, num_clients=num_clients)\n"
    )
    first = run_function_from_file(str(loader), "get_data", num_clients=3)
    second = run_function_from_file(str(loader), "get_data", num_clients=3)
    assert first == [3]
    assert second is first


def test_build_arguments_may_be_named_key_or_build():
    # Loaders forward the dataset kwargs of a config, whatever their names.
    build = RecordingBuild()
    cache_per_process("named-like-params", build, key="k", build="b")
    assert build.calls == [{"key": "k", "build": "b"}]


def test_arguments_that_cannot_be_compared_rebuild_instead_of_failing():
    build = RecordingBuild()
    cache_per_process("uncomparable", build, weights=np.ones(3))
    cache_per_process("uncomparable", build, weights=np.ones(3))
    assert len(build.calls) == 2


@pytest.mark.parametrize(
    "partition, kwargs",
    [
        (iid_partition, {}),
        (class_noniid_partition, {"seed": 3}),
        (dirichlet_noniid_partition, {"alpha2": 0.5, "seed": 3}),
    ],
)
def test_cached_partition_matches_a_fresh_one_for_every_client(partition, kwargs):
    """The partition functions are deterministic, so caching them is safe."""
    dataset = _Labeled()

    def build(num_clients, **kwargs):
        return partition(dataset, num_clients, **kwargs)

    for client_id in range(5):
        np.random.seed(100 + client_id)  # each client starts from another state
        cached = cache_per_process(
            ("real-partition", partition.__name__), build, num_clients=5, **kwargs
        )[client_id]
        after_cached = np.random.rand()

        np.random.seed(100 + client_id)
        fresh = partition(dataset, 5, **kwargs)[client_id]
        after_fresh = np.random.rand()

        assert torch.equal(cached.data_input, fresh.data_input)
        assert torch.equal(cached.data_label, fresh.data_label)
        assert after_cached == after_fresh
