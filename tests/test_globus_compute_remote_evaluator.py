"""Tests for appfl.comm.globus_compute.remote_evaluator without a live endpoint:
the "endpoint" is a thread in this process, so the shipped task, the endpoint-side
cache and the config validation all run for real."""

import concurrent.futures

import pytest
from omegaconf import OmegaConf

from appfl.comm.globus_compute import remote_evaluator
from appfl.comm.globus_compute.remote import GlobusComputeRemote
from appfl.comm.globus_compute.remote_evaluator import (
    RemoteEvaluator,
    create_remote_evaluator,
    evaluate_on_endpoint,
)
from appfl.decentralized.metrics import Meter


_ENDPOINT_ID = "00000000-0000-0000-0000-000000000000"

_EVALUATOR_SOURCE = '''
BUILT = []


def get_evaluator(offset=0.0):
    BUILT.append(offset)
    return lambda x: x + offset


def get_kwargs_evaluator():
    return lambda x, **kwargs: kwargs["eval_idx"]
'''


class _FakeExecutor:
    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit(self, fn, *args, **kwargs):
        return self._pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        self._pool.shutdown(wait=False)


@pytest.fixture
def evaluator_file(tmp_path):
    path = tmp_path / "fake_evaluator.py"
    path.write_text(_EVALUATOR_SOURCE)
    return str(path)


@pytest.fixture(autouse=True)
def _clear_endpoint_cache():
    remote_evaluator._evaluators.clear()
    yield
    remote_evaluator._evaluators.clear()


@pytest.fixture
def built_remotes(monkeypatch):
    """Make create_remote_evaluator build its remote on a fake executor."""
    built = []

    def factory(**kwargs):
        built.append(kwargs)
        return GlobusComputeRemote(executor=_FakeExecutor(), **kwargs)

    monkeypatch.setattr(remote_evaluator, "GlobusComputeRemote", factory)
    return built


def _configs(evaluator_file, **remote):
    # The local keys deliberately differ from the remote ones: the remote block must not
    # read them.
    return {
        "evaluator_path": "./local/only.py",
        "evaluator_name": "get_kwargs_evaluator",
        "evaluator_kwargs": {"offset": 99.0},
        "remote": {
            "backend": "globus_compute",
            "endpoint_id": _ENDPOINT_ID,
            "evaluator_path": evaluator_file,
            "evaluator_name": "get_evaluator",
            "evaluator_kwargs": {"offset": 1.0},
            **remote,
        },
    }


def _context(eval_idx):
    return {"agent_id": "agent-0", "round_idx": eval_idx + 3, "eval_idx": eval_idx}


def test_endpoint_builds_evaluator_once_per_arguments(evaluator_file):
    assert evaluate_on_endpoint(evaluator_file, "get_evaluator", {}, 1.0, _context(0))[0] == 1.0
    assert evaluate_on_endpoint(evaluator_file, "get_evaluator", {}, 2.0, _context(1))[0] == 2.0
    assert evaluate_on_endpoint(
        evaluator_file, "get_evaluator", {"offset": 5.0}, 1.0, _context(0)
    )[0] == 6.0
    assert len(remote_evaluator._evaluators) == 2


def test_endpoint_forwards_context_only_to_kwargs_evaluators(evaluator_file):
    value, seconds = evaluate_on_endpoint(
        evaluator_file, "get_kwargs_evaluator", {}, 0.5, _context(7)
    )
    assert value == 7.0 and seconds >= 0.0
    # A plain `f(x)` evaluator is not passed the context, so this does not raise.
    assert evaluate_on_endpoint(evaluator_file, "get_evaluator", {}, 0.5, _context(7))[0] == 0.5


def test_remote_evaluator_forwards_agent_context_and_meters_overhead(evaluator_file):
    meter = Meter()
    evaluator = RemoteEvaluator(
        GlobusComputeRemote(_ENDPOINT_ID, executor=_FakeExecutor()),
        evaluator_file,
        "get_kwargs_evaluator",
        meter=meter,
    )
    # The index comes from the caller (the agent), not from a counter in the wrapper.
    assert [evaluator(0.0, **_context(k)) for k in (4, 2, 9)] == [4.0, 2.0, 9.0]
    assert meter.remote_overhead_seconds >= 0.0
    evaluator.close()


def test_create_from_config_runs_remotely(evaluator_file, built_remotes):
    evaluator = create_remote_evaluator(_configs(evaluator_file, timeout=30))
    assert evaluator(2.0) == 3.0
    assert built_remotes[0]["endpoint_id"] == _ENDPOINT_ID
    assert built_remotes[0]["timeout"] == 30.0
    assert built_remotes[0]["compute_token"] is None
    evaluator.close()


def test_local_keys_beside_the_block_are_ignored(evaluator_file, built_remotes):
    evaluator = create_remote_evaluator(_configs(evaluator_file))
    assert evaluator.evaluator_path == evaluator_file
    assert evaluator.evaluator_name == "get_evaluator"
    assert evaluator.evaluator_kwargs == {"offset": 1.0}


def test_evaluator_path_is_required_in_the_block(evaluator_file, built_remotes):
    configs = _configs(evaluator_file)
    del configs["remote"]["evaluator_path"]
    with pytest.raises(ValueError, match="evaluator_path"):
        create_remote_evaluator(configs)


def test_missing_evaluator_kwargs_defaults_to_empty(evaluator_file, built_remotes):
    configs = _configs(evaluator_file)
    del configs["remote"]["evaluator_kwargs"]
    evaluator = create_remote_evaluator(configs)
    assert evaluator.evaluator_kwargs == {}
    assert evaluator(1.0) == 1.0  # offset defaults to 0


def test_missing_evaluator_name_uses_last_function_and_warns(
    evaluator_file, built_remotes, caplog
):
    configs = _configs(evaluator_file, evaluator_kwargs={})
    del configs["remote"]["evaluator_name"]
    with caplog.at_level("WARNING"):
        evaluator = create_remote_evaluator(configs)
    assert "last function" in caplog.text
    # The last function in the file is get_kwargs_evaluator, resolved on the endpoint.
    assert evaluator(0.0, eval_idx=5) == 5.0


def test_endpoint_resolves_last_function_when_name_is_none(evaluator_file):
    assert evaluate_on_endpoint(evaluator_file, None, {}, 0.0, _context(4))[0] == 4.0


def test_remote_only_config_needs_no_local_keys(evaluator_file, built_remotes):
    configs = {
        "remote": {
            "endpoint_id": _ENDPOINT_ID,
            "evaluator_path": evaluator_file,
            "evaluator_name": "get_evaluator",
            "evaluator_kwargs": {"offset": 2.0},
        }
    }
    evaluator = create_remote_evaluator(configs)
    assert evaluator(1.0) == 3.0
    evaluator.close()


def test_interpolated_config_is_resolved(evaluator_file, built_remotes):
    node_config = OmegaConf.create(
        {
            "space_configs": {"space_kwargs": {"agent_index": 3}},
            "evaluator_configs": _configs(
                evaluator_file,
                evaluator_kwargs={"offset": "${space_configs.space_kwargs.agent_index}"},
            ),
        }
    )
    evaluator = create_remote_evaluator(node_config.evaluator_configs)
    assert evaluator.evaluator_kwargs == {"offset": 3}
    assert evaluator(1.0) == 4.0


def test_tokens_are_read_from_named_env_vars(evaluator_file, built_remotes, monkeypatch):
    monkeypatch.setenv("TEST_GC_COMPUTE", "compute-placeholder")
    monkeypatch.setenv("TEST_GC_OPENID", "openid-placeholder")
    create_remote_evaluator(
        _configs(
            evaluator_file,
            compute_token_env="TEST_GC_COMPUTE",
            openid_token_env="TEST_GC_OPENID",
        )
    )
    assert built_remotes[0]["compute_token"] == "compute-placeholder"
    assert built_remotes[0]["openid_token"] == "openid-placeholder"


def test_unset_token_env_var_is_an_error(evaluator_file, built_remotes, monkeypatch):
    monkeypatch.delenv("TEST_GC_UNSET", raising=False)
    with pytest.raises(ValueError, match="TEST_GC_UNSET"):
        create_remote_evaluator(
            _configs(evaluator_file, compute_token_env="TEST_GC_UNSET")
        )


@pytest.mark.parametrize(
    "remote, match",
    [
        ({"endpoint_id": None}, "endpoint_id"),
        ({"backend": "ray"}, "backend"),
        ({"endpoint": _ENDPOINT_ID}, "unknown"),
    ],
)
def test_invalid_remote_block_is_rejected(evaluator_file, built_remotes, remote, match):
    with pytest.raises(ValueError, match=match):
        create_remote_evaluator(_configs(evaluator_file, **remote))
