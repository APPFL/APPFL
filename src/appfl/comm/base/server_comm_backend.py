from abc import ABC, abstractmethod
from appfl.logger import ServerAgentFileLogger
from appfl.config import ClientAgentConfig, ServerAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Tuple, Any


class _PrefixLoggerAdapter:
    """
    Thin logger wrapper that prefixes every message with a fixed tag (e.g.,
    ``[tes_backend]`` for a backend, or ``[hybrid]`` for the driver), so log lines
    from different components are distinguishable in a hybrid federation. Unknown
    attributes are forwarded to the wrapped logger.

    If the wrapped logger is itself a ``_PrefixLoggerAdapter`` (e.g., the driver's
    already-prefixed logger passed down to a backend), it is unwrapped first so
    prefixes do not stack.
    """

    def __init__(self, logger, prefix: str):
        self._logger = (
            logger._logger if isinstance(logger, _PrefixLoggerAdapter) else logger
        )
        self._prefix = prefix

    def info(self, msg, *args, **kwargs):
        self._logger.info(f"{self._prefix} {msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(f"{self._prefix} {msg}", *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(f"{self._prefix} {msg}", *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(f"{self._prefix} {msg}", *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class ServerCommBackend(ABC):
    """
    Transport-specific adapter for a *server-driven* communication protocol
    (e.g., Globus Compute, TES). A backend owns the subset of clients that use
    its transport and knows how to (a) transfer a model to those clients,
    (b) submit a task and obtain a ``concurrent.futures.Future``-compatible
    handle, (c) parse a completed result, and (d) cancel/shutdown.

    The generic :class:`~appfl.comm.base.server_driven_communicator.ServerDrivenCommunicator`
    drives one or more backends, owning all cross-transport task bookkeeping and
    the unified ``as_completed`` receive loop. This separation is what enables a
    single federation to mix transports (a "hybrid" federation).

    :param `comm_type`: String identifying the transport (used for S3 temp dirs, logs).
    :param `server_agent_config`: The server agent configuration.
    :param `client_agent_configs`: The subset of client configurations served by this backend.
    :param `experiment_id`: Shared experiment id assigned by the driver.
    :param [Optional] `logger`: Logger object shared with the driver.
    """

    def __init__(
        self,
        comm_type: str,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        experiment_id: str,
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = comm_type
        self.server_agent_config = server_agent_config
        self.client_agent_configs = client_agent_configs
        self.experiment_id = experiment_id
        # Prefix this backend's log lines with its transport name so they are
        # distinguishable from other backends' lines in a hybrid federation.
        self.logger = (
            _PrefixLoggerAdapter(logger, f"[{comm_type}_backend]")
            if logger is not None
            else logger
        )
        self._check_client_comm_type(client_agent_configs, comm_type)

    @staticmethod
    def _check_client_comm_type(client_agent_configs, comm_type: str):
        """
        Guard against routing mistakes: if a client config explicitly declares a
        ``comm_configs.comm_type``, it must match the backend it was handed to.
        Clients that omit ``comm_type`` (relying on auto-inference) are allowed.
        """
        for client_config in client_agent_configs:
            comm_configs = getattr(client_config, "comm_configs", None)
            declared = getattr(comm_configs, "comm_type", None)
            if declared is not None and str(declared) != comm_type:
                client_id = getattr(client_config, "client_id", "<unknown>")
                raise ValueError(
                    f"Client '{client_id}' declares comm_type='{declared}' but was "
                    f"routed to the '{comm_type}' backend."
                )

    @property
    @abstractmethod
    def client_ids(self) -> List[str]:
        """Return the list of client ids served by this backend (submission order)."""
        ...

    @abstractmethod
    def prepare_model(self, model: Optional[Union[Dict, OrderedDict, bytes]]) -> Any:
        """
        Transport-specific one-time preparation of a shared model before it is
        submitted to (potentially many) clients. For transports that upload the
        model to intermediate storage (S3/ProxyStore/file storage), this uploads
        it once and returns a lightweight handle/reference to embed per task.
        Called once per backend by ``send_task_to_all_clients`` to avoid
        re-uploading the same model for every client.

        :param `model`: The model to prepare (or ``None``).
        :return: A transport-specific object to pass to :meth:`submit_task`.
        """
        ...

    @abstractmethod
    def submit_task(
        self,
        client_id: str,
        task_name: str,
        prepared_model: Any = None,
        metadata: Optional[Dict] = None,
        need_model_response: bool = False,
    ) -> Tuple[str, Any]:
        """
        Submit a task to a single client and return a handle for its result.

        :param `client_id`: The client to submit the task to.
        :param `task_name`: Name of the task to execute on the client.
        :param `prepared_model`: The output of :meth:`prepare_model` (or ``None``).
        :param `metadata`: Additional metadata for the task.
        :param `need_model_response`: Whether the task returns a model (may require
            pre-signed upload URLs).
        :return `task_id`: A unique id for the submitted task.
        :return `future`: A ``concurrent.futures.Future``-compatible object whose
            ``result()`` yields the client's raw result.
        """
        ...

    @abstractmethod
    def process_result(self, future: Any) -> Tuple[Any, Dict]:
        """
        Extract and parse the result of a completed task future for this transport
        (including any storage download of the returned model).

        :param `future`: A future previously returned by :meth:`submit_task`.
        :return `model`: The model parameters returned from the client.
        :return `metadata`: The metadata returned from the client.
        """
        ...

    @abstractmethod
    def cancel_task(self, task_id: str, future: Any, client_id: str):
        """Cancel a single on-the-fly task."""
        ...

    @abstractmethod
    def shutdown(self):
        """Cancel remaining work and release all transport resources."""
        ...
