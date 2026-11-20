"""Tests for appfl.comm.globus_compute.remote without a live endpoint: the
executor is injected (or the SDK classes are monkeypatched), and the endpoint
id is a placeholder."""

import sys
import time
import types
import pytest
import concurrent.futures
from appfl.comm.globus_compute import remote
from appfl.comm.globus_compute.remote import GlobusComputeRemote, get_compute_client
from appfl.comm.globus_compute.utils import client as client_utils


_ENDPOINT_ID = "00000000-0000-0000-0000-000000000000"


class _FakeExecutor:
    """Thread-backed stand-in for a Globus Compute Executor."""

    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.shutdown_calls = []

    def submit(self, fn, *args, **kwargs):
        return self._pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))
        self._pool.shutdown(wait=False)


def _add(a, b, scale=1):
    return (a + b) * scale


def _fail():
    raise ValueError("remote failure")


def _sleep(seconds):
    time.sleep(seconds)


@pytest.fixture
def fake():
    return _FakeExecutor()


@pytest.fixture(autouse=True)
def _clear_client_cache():
    remote._clients.clear()
    yield
    remote._clients.clear()


def test_run_returns_result_with_args_and_kwargs(fake):
    with GlobusComputeRemote(_ENDPOINT_ID, executor=fake) as gc:
        assert gc.run(_add, 1, 2, scale=3) == 9


def test_submit_returns_future(fake):
    with GlobusComputeRemote(_ENDPOINT_ID, executor=fake) as gc:
        assert gc.submit(_add, 2, 2).result(timeout=5) == 4


def test_remote_exception_is_reraised_without_closing(fake):
    with GlobusComputeRemote(_ENDPOINT_ID, executor=fake) as gc:
        with pytest.raises(ValueError, match="remote failure"):
            gc.run(_fail)
        assert fake.shutdown_calls == []


def test_timeout_raises_builtin_timeout_error_and_closes(fake):
    gc = GlobusComputeRemote(_ENDPOINT_ID, timeout=0.1, executor=fake)
    with pytest.raises(TimeoutError, match=_ENDPOINT_ID):
        gc.run(_sleep, 1.0)
    assert fake.shutdown_calls == [(False, True)]
    with pytest.raises(RuntimeError):
        gc.submit(_add, 1, 1)


def _raise_timeout():
    raise TimeoutError("raised by the task itself")


def test_task_raising_timeout_error_passes_through_without_closing(fake):
    with GlobusComputeRemote(_ENDPOINT_ID, timeout=5, executor=fake) as gc:
        with pytest.raises(TimeoutError, match="by the task itself"):
            gc.run(_raise_timeout)
        assert fake.shutdown_calls == []
        assert gc.run(_add, 1, 1) == 2


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_timeout_is_rejected(fake, bad):
    with pytest.raises(ValueError):
        GlobusComputeRemote(_ENDPOINT_ID, timeout=bad, executor=fake)


def test_close_is_idempotent_and_blocks_new_tasks(fake):
    gc = GlobusComputeRemote(_ENDPOINT_ID, executor=fake)
    gc.close()
    gc.close()
    assert fake.shutdown_calls == [(False, True)]
    with pytest.raises(RuntimeError):
        gc.submit(_add, 1, 1)


def test_context_manager_closes(fake):
    with GlobusComputeRemote(_ENDPOINT_ID, executor=fake):
        pass
    assert fake.shutdown_calls == [(False, True)]


def test_executor_is_bound_to_endpoint(monkeypatch):
    built = {}

    def fake_executor(**kwargs):
        built.update(kwargs)
        return _FakeExecutor()

    monkeypatch.setattr(remote, "Executor", fake_executor)
    monkeypatch.setattr(remote, "Client", lambda **kwargs: "client")
    uec = {"account": "placeholder", "walltime": "01:00:00"}
    with GlobusComputeRemote(_ENDPOINT_ID, user_endpoint_config=uec):
        pass
    assert built["endpoint_id"] == _ENDPOINT_ID
    assert built["client"] == "client"
    assert built["user_endpoint_config"] == uec
    assert built["resource_specification"] is None


@pytest.mark.parametrize("tokens", [{"compute_token": "c"}, {"openid_token": "o"}])
def test_tokens_must_come_in_pairs(tokens):
    with pytest.raises(ValueError):
        get_compute_client(**tokens)


def test_tokens_build_token_client_and_are_cached(monkeypatch):
    built = []

    def fake_create_token_client(compute_token, openid_token, **kwargs):
        built.append((compute_token, openid_token))
        return {"tokens": (compute_token, openid_token), **kwargs}

    monkeypatch.setattr(remote, "create_token_client", fake_create_token_client)

    client = get_compute_client("compute-placeholder", "openid-placeholder")
    assert client["tokens"] == ("compute-placeholder", "openid-placeholder")
    assert "code_serialization_strategy" in client

    assert get_compute_client("compute-placeholder", "openid-placeholder") is client
    assert len(built) == 1
    get_compute_client("compute-other", "openid-other")
    assert len(built) == 2


def test_no_tokens_uses_default_login(monkeypatch):
    monkeypatch.setattr(
        remote,
        "create_token_client",
        lambda *args, **kwargs: pytest.fail("no tokens should mean no token client"),
    )
    monkeypatch.setattr(remote, "Client", lambda **kwargs: kwargs)
    client = get_compute_client()
    assert "authorizer" not in client
    assert get_compute_client() is client


def test_token_client_uses_authorizer_when_the_sdk_accepts_it(monkeypatch):
    class FakeClient:
        def __init__(self, authorizer=None, code_serialization_strategy=None):
            self.authorizer = authorizer
            self.code_serialization_strategy = code_serialization_strategy

    monkeypatch.setattr(client_utils, "Client", FakeClient)
    client = client_utils.create_token_client(
        "compute-placeholder", "openid-placeholder", code_serialization_strategy="code"
    )
    assert client.authorizer.access_token == "compute-placeholder"
    assert client.code_serialization_strategy == "code"


def test_token_client_falls_back_to_login_manager_on_old_sdks(monkeypatch):
    managers = []

    class FakeLoginManager:
        def __init__(self, authorizers):
            self.authorizers = authorizers
            self.logged_in = False
            managers.append(self)

        def ensure_logged_in(self):
            self.logged_in = True

    class FakeOldClient:
        def __init__(self, login_manager=None, code_serialization_strategy=None):
            self.login_manager = login_manager

    module = types.ModuleType("globus_compute_sdk.sdk.login_manager")
    module.AuthorizerLoginManager = FakeLoginManager
    monkeypatch.setitem(sys.modules, "globus_compute_sdk.sdk.login_manager", module)
    monkeypatch.setattr(client_utils, "Client", FakeOldClient)

    client = client_utils.create_token_client(
        "compute-placeholder", "openid-placeholder"
    )
    assert client.login_manager is managers[0]
    assert managers[0].logged_in
    tokens = {a.access_token for a in managers[0].authorizers.values()}
    assert tokens == {"compute-placeholder", "openid-placeholder"}
