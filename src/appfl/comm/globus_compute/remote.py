import atexit
import threading
import concurrent.futures
from typing import Any, Callable, Dict, Optional, Tuple
from globus_sdk import AccessTokenAuthorizer
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager

_clients: Dict[Tuple[Optional[str], Optional[str]], Client] = {}
_clients_lock = threading.Lock()


def get_compute_client(
    compute_token: Optional[str] = None,
    openid_token: Optional[str] = None,
) -> Client:
    """
    Return a Globus Compute client, cached per process and per token pair, so that
    several nodes in one process share a single login.

    :param `compute_token`: Globus Compute access token; must be given with `openid_token`.
    :param `openid_token`: Globus Auth (OpenID) access token; must be given with `compute_token`.
        Without tokens, the SDK's default login is used: the cached login, or a confidential
        client from the `GLOBUS_COMPUTE_CLIENT_ID`/`GLOBUS_COMPUTE_CLIENT_SECRET` env vars.
    """
    if (compute_token is None) != (openid_token is None):
        raise ValueError(
            "compute_token and openid_token must be provided together, or not at all."
        )
    key = (compute_token, openid_token)
    with _clients_lock:
        if key not in _clients:
            if compute_token is None:
                client = Client(code_serialization_strategy=CombinedCode())
            else:
                login_manager = AuthorizerLoginManager(
                    authorizers={
                        ComputeScopes.resource_server: AccessTokenAuthorizer(
                            compute_token
                        ),
                        AuthScopes.resource_server: AccessTokenAuthorizer(
                            openid_token
                        ),
                    }
                )
                login_manager.ensure_logged_in()
                client = Client(
                    login_manager=login_manager,
                    code_serialization_strategy=CombinedCode(),
                )
            _clients[key] = client
        return _clients[key]


class GlobusComputeRemote:
    """
    A Globus Compute executor bound to one endpoint.

    The submitted function is serialized with its source code, so it should be a thin
    top-level function that imports the heavy logic from the endpoint's own installed
    packages (the endpoint must have a matching `appfl` installed).

    :param `endpoint_id`: The Globus Compute endpoint to run tasks on.
    :param [Optional] `timeout`: Seconds :meth:`run` waits for a result before raising
        `TimeoutError`; `None` waits forever (tasks for an offline endpoint stay queued).
    :param [Optional] `compute_token`: See :func:`get_compute_client`.
    :param [Optional] `openid_token`: See :func:`get_compute_client`.
    :param [Optional] `user_endpoint_config`: Values for a multi-user endpoint's config
        template (e.g., account, queue, walltime on an HPC endpoint).
    :param [Optional] `resource_specification`: Per-task resource request (MPI endpoints).
    :param [Optional] `executor`: A pre-built executor to use instead of a Globus Compute
        one; it takes ownership, and is mainly for testing.
    """

    def __init__(
        self,
        endpoint_id: str,
        timeout: Optional[float] = None,
        compute_token: Optional[str] = None,
        openid_token: Optional[str] = None,
        user_endpoint_config: Optional[Dict[str, Any]] = None,
        resource_specification: Optional[Dict[str, Any]] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ):
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}.")
        self.endpoint_id = str(endpoint_id)
        self.timeout = timeout
        if executor is None:
            executor = Executor(
                endpoint_id=self.endpoint_id,
                client=get_compute_client(compute_token, openid_token),
                user_endpoint_config=user_endpoint_config,
                resource_specification=resource_specification,
            )
        self._executor = executor
        self._closed = False
        # Safety net: an unclosed executor keeps background threads alive at exit.
        atexit.register(self.close)

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit `fn(*args, **kwargs)` to the endpoint and return its future."""
        if self._closed:
            raise RuntimeError(f"Remote for endpoint {self.endpoint_id} is closed.")
        return self._executor.submit(fn, *args, **kwargs)

    def run(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Run `fn(*args, **kwargs)` on the endpoint and return its result, re-raising
        any exception from the remote task.
        """
        future = self.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except BaseException as exc:
            if future.done():
                raise
            self.close()
            if isinstance(exc, concurrent.futures.TimeoutError):
                raise TimeoutError(
                    f"Task {getattr(fn, '__name__', fn)} on endpoint {self.endpoint_id} "
                    f"did not finish within {self.timeout} seconds."
                ) from None
            raise

    def close(self):
        """Cancel pending tasks and shut down the executor; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
