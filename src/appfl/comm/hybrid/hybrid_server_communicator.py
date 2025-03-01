import time
from typing import (
    List,
    Optional,
    Union,
    Dict,
    Any,
    Tuple,
)

from collections import OrderedDict

from appfl.comm.base import BaseServerCommunicator
from appfl.config import ServerAgentConfig, ClientAgentConfig
from appfl.logger import ServerAgentFileLogger

from appfl.comm.ray import RayServerCommunicator
from appfl.comm.globus_compute import GlobusComputeServerCommunicator
import threading
import queue


class HybridServerCommunicator(BaseServerCommunicator):
    """
    A hybrid communicator that delegates to either a RayServerCommunicator or a
    GlobusComputeServerCommunicator on a per-client basis, depending on the 'comm_type'
    specified in each ClientAgentConfig.

    If a client's comm_type is 'ray', it is handled by the Ray communicator;
    if it's 'globus_compute', it is handled by the Globus communicator.

    All server communicator interface methods are proxied to the respective
    sub-communicator(s). For example, `send_task_to_all_clients` will send tasks
    to *both* the Ray-based clients and Globus-based clients.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        """
        Constructor separates client configurations into Ray vs Globus,
        then instantiates two sub-communicators (if needed).

        :param server_agent_config: ServerAgentConfig object
        :param client_agent_configs: List of ClientAgentConfig objects
        :param logger: Optional logger
        :param kwargs: Additional keyword arguments; may include tokens for Globus, etc.
        """
        self.comm_type = "hybrid"

        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)

        self.ray_client_configs = []
        self.globus_client_configs = []

        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()

        for cfg in client_agent_configs:
            ctype = getattr(cfg, "comm_type", None)
            if ctype == "ray":
                self.ray_client_configs.append(cfg)
            elif ctype == "globus_compute":
                self.globus_client_configs.append(cfg)
            else:
                raise ValueError(
                    f"Hybrid communicator expects either 'ray' or 'globus_compute', got {ctype}"
                )

        self.ray_communicator = (
            RayServerCommunicator(
                server_agent_config,
                self.ray_client_configs,
                logger=logger,
                **kwargs,
            )
            if self.ray_client_configs
            else None
        )

        self.globus_communicator = (
            GlobusComputeServerCommunicator(
                server_agent_config,
                self.globus_client_configs,
                logger=logger,
                **kwargs,
            )
            if self.globus_client_configs
            else None
        )

        if self.ray_communicator:
            self._ray_thread = threading.Thread(
                target=self._poll_ray, daemon=True
            )
            self._ray_thread.start()

        if self.globus_communicator:
            self._globus_thread = threading.Thread(
                target=self._poll_globus, daemon=True
            )
            self._globus_thread.start()

    def _poll_ray(self):
        while not self._stop_event.is_set():
            try:
                cid, model, meta = self.ray_communicator.recv_result_from_one_client()
                self._result_queue.put((cid, model, meta))
            except AssertionError:
                pass
            except Exception as e:
                pass
            time.sleep(0.2)

    def _poll_globus(self):
        while not self._stop_event.is_set():
            try:
                cid, model, meta = self.globus_communicator.recv_result_from_one_client()
                self._result_queue.put((cid, model, meta))
            except AssertionError:
                pass
            except Exception as e:
                pass
            time.sleep(0.2)

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """
        Dispatches the same `task_name` to all Ray clients and all GlobusCompute clients.

        :param task_name: The name of the task to send
        :param model: Optional model data
        :param metadata: Additional metadata (dict or list of dicts)
        :param need_model_response: Whether we expect clients to send back a (large) model
        """
        if self.ray_communicator is not None and self.ray_client_configs:
            self.ray_communicator.send_task_to_all_clients(
                task_name,
                model=model,
                metadata=metadata,
                need_model_response=need_model_response,
            )

        if self.globus_communicator is not None and self.globus_client_configs:
            self.globus_communicator.send_task_to_all_clients(
                task_name,
                model=model,
                metadata=metadata,
                need_model_response=need_model_response,
            )

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a task to a single client, identified by client_id.

        We decide which sub-communicator to delegate to by checking if
        that client_id is in the Ray communicator or the Globus communicator.
        """
        if self.ray_communicator and (client_id in self.ray_communicator.client_actors):
            self.ray_communicator.send_task_to_one_client(
                client_id,
                task_name,
                model=model,
                metadata=metadata,
                need_model_response=need_model_response,
            )
        elif self.globus_communicator and (client_id in self.globus_communicator.client_endpoints):
            self.globus_communicator.send_task_to_one_client(
                client_id,
                task_name,
                model=model,
                metadata=metadata,
                need_model_response=need_model_response,
            )
        else:
            raise ValueError(f"Unknown client_id {client_id} in hybrid communicator.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Collect results from all sub-communicator tasks.

        :return: (client_results, client_metadata), each a dict keyed by client_id
        """
        all_client_results = {}
        all_client_metadata = {}

        if self.ray_communicator and self.ray_client_configs:
            ray_results, ray_metadata = self.ray_communicator.recv_result_from_all_clients()
            all_client_results.update(ray_results)
            all_client_metadata.update(ray_metadata)

        if self.globus_communicator and self.globus_client_configs:
            gc_results, gc_metadata = self.globus_communicator.recv_result_from_all_clients()
            all_client_results.update(gc_results)
            all_client_metadata.update(gc_metadata)

        return all_client_results, all_client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Return the first completed result from *either* sub-communicator.
        We poll both Ray tasks and Globus tasks, returning whichever finishes first.

        :return: (client_id, client_model, client_metadata)
        """
        return self._result_queue.get()

    def shutdown_all_clients(self):
        """
        Shut down both sets of clients and do any necessary cleanup (S3, proxystore, etc.).
        """
        self._stop_event.set()
        if self.ray_communicator:
            self.ray_communicator.shutdown_all_clients()
        if self.globus_communicator:
            self.globus_communicator.shutdown_all_clients()

    def cancel_all_tasks(self):
        """
        Cancel all on-the-fly tasks in both sub-communicators.
        """
        if self.ray_communicator:
            self.ray_communicator.cancel_all_tasks()
        if self.globus_communicator:
            self.globus_communicator.cancel_all_tasks()
