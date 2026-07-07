import time
from concurrent.futures import as_completed
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base.server_comm_backend import ServerCommBackend, _PrefixLoggerAdapter
from appfl.comm.base.base_server_communicator import BaseServerCommunicator
from appfl.config import ClientAgentConfig, ServerAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Tuple, Any


class ServerDrivenCommunicator(BaseServerCommunicator):
    """
    Generic, transport-agnostic driver for *server-driven* federated learning.

    The driver owns all cross-transport task bookkeeping (``executing_tasks``,
    ``executing_task_futs``) and a single unified ``as_completed`` receive loop.
    The actual transport work is delegated to one or more
    :class:`~appfl.comm.base.server_comm_backend.ServerCommBackend` instances,
    each owning the subset of clients that use its transport.

    Because a single driver can hold multiple backends and every backend returns
    ``concurrent.futures.Future``-compatible handles, one federation can mix
    transports (see :class:`~appfl.comm.hybrid.HybridServerCommunicator`).
    Single-transport communicators (Globus Compute, TES) are thin subclasses that
    register a single backend.

    Subclasses implement :meth:`_build_backends` to construct and register their
    backend(s) via :meth:`_add_backend`.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        if not hasattr(self, "comm_type"):
            self.comm_type = "server_driven"
        super().__init__(
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            logger=logger,
            **kwargs,
        )
        # Prefix the driver's own log lines with a generic [Communicator] tag.
        # Backends re-wrap this logger with their own tag; the adapter unwraps first
        # so backend lines stay [<comm_type>_backend] rather than stacking prefixes.
        if self.logger is not None:
            self.logger = _PrefixLoggerAdapter(self.logger, "[Communicator]")
        # Registry of transport backends and per-client routing.
        self.backends: List[ServerCommBackend] = []
        self.client_backend: Dict[str, ServerCommBackend] = {}
        self.client_ids: List[str] = []
        # Map each in-flight future to the backend that produced it.
        self._fut_backend: Dict[Any, ServerCommBackend] = {}
        self._build_backends(**kwargs)

    def _init_storage(self, server_agent_config):
        """
        Storage (S3 / ProxyStore / file storage) is owned by each backend, so the
        driver itself performs no model transfer and initializes no storage.
        """
        self.use_s3bucket = False
        self.use_proxystore = False
        self.proxystore = None

    def _build_backends(self, **kwargs):
        """
        Construct the transport backend(s) and register them via
        :meth:`_add_backend`. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _add_backend(self, backend: ServerCommBackend):
        """Register a backend and route its clients to it."""
        self.backends.append(backend)
        for client_id in backend.client_ids:
            assert client_id not in self.client_backend, (
                f"Client ID {client_id} is served by more than one backend."
            )
            self.client_backend[client_id] = backend
            self.client_ids.append(client_id)

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        # Prepare (e.g., upload) the shared model at most once per backend.
        prepared_by_backend: Dict[int, Any] = {}
        for i, client_id in enumerate(self.client_ids):
            backend = self.client_backend[client_id]
            if id(backend) not in prepared_by_backend:
                prepared_by_backend[id(backend)] = (
                    backend.prepare_model(model) if model is not None else None
                )
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            task_id, future = backend.submit_task(
                client_id,
                task_name,
                prepared_by_backend[id(backend)],
                client_metadata,
                need_model_response,
            )
            self._register_task(task_id, future, client_id, task_name)
            self._fut_backend[future] = backend
            self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        assert client_id in self.client_backend, (
            f"Client ID {client_id} is not managed by this communicator."
        )
        backend = self.client_backend[client_id]
        prepared_model = backend.prepare_model(model) if model is not None else None
        task_id, future = backend.submit_task(
            client_id,
            task_name,
            prepared_model,
            metadata,
            need_model_response,
        )
        self._register_task(task_id, future, client_id, task_name)
        self._fut_backend[future] = backend
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        client_results, client_metadata = {}, {}
        while len(self.executing_task_futs):
            fut = next(as_completed(list(self.executing_task_futs)))
            client_id, client_model, client_metadata_local = self._process_future(fut)
            client_results[client_id] = client_model
            client_metadata[client_id] = client_metadata_local
        return client_results, client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        assert len(self.executing_task_futs), "There is no active client running tasks."
        fut = next(as_completed(list(self.executing_task_futs)))
        return self._process_future(fut)

    def _process_future(self, fut: Any) -> Tuple[str, Any, Dict]:
        """Parse a single completed future via its owning backend and update bookkeeping."""
        task_id = self.executing_task_futs[fut]
        client_id = self.executing_tasks[task_id].client_id
        backend = self._fut_backend[fut]
        try:
            client_model, client_metadata = backend.process_result(fut)
            client_metadata = self._check_deprecation(client_id, client_metadata)
            # Record the status of the finished task.
            client_log = client_metadata.get("log", {})
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].log = client_log
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
            )
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(fut)
            self._fut_backend.pop(fut)
        except Exception as e:
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e
        return client_id, client_model, client_metadata

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks across all backends."""
        for fut in list(self.executing_task_futs):
            task_id = self.executing_task_futs[fut]
            client_id = self.executing_tasks[task_id].client_id
            backend = self._fut_backend.get(fut)
            if backend is not None:
                backend.cancel_task(task_id, fut, client_id)
        self.executing_task_futs = {}
        self.executing_tasks = {}
        self._fut_backend = {}

    def shutdown_all_clients(self):
        """Shutdown every backend and release resources."""
        self.logger.info("Shutting down all clients......")
        for backend in self.backends:
            backend.shutdown()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )
