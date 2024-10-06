import time
import json
import logging
import threading
from mpi4py import MPI
from omegaconf import OmegaConf
from typing import Optional, Dict
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.logger import ServerAgentFileLogger
from .serializer import byte_to_request, response_to_byte, model_to_byte
from .config import MPITask, MPITaskRequest, MPITaskResponse, MPIServerStatus

class MPIServerCommunicator:
    def __init__(
        self, 
        comm,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()
        self._global_model_futures: Dict[int, Future] = {}
        self._meta_data_futures: Dict[int, Future] = {}

    def serve(self):
        """
        Start the server to serve the clients.
        """
        self.logger.info(f"Server starting...")
        status = MPI.Status()
        while not self.server_agent.server_terminated():
            time.sleep(0.1)
            msg_flag = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if msg_flag:
                source = status.Get_source()
                tag = status.Get_tag()
                count = status.Get_count(MPI.BYTE)
                request_buffer = bytearray(count)
                self.comm.Recv(request_buffer, source=source, tag=tag)
                request = byte_to_request(request_buffer)
                response = self._request_handler(client_id=source, request_tag=tag, request=request)
                if response is not None:
                    response_bytes = response_to_byte(response)
                    self.comm.Send(response_bytes, dest=source, tag=source)
        self.logger.info(f"Server terminated.")

    def _request_handler(
        self, 
        client_id: int,
        request_tag: int, 
        request: MPITaskRequest, 
    ) -> Optional[MPITaskResponse]:
        """
        Handle the request from the clients.
        :param `request`: the request from the clients
        :param `request_tag`: the tag of the request
        :return `response`: the response to the clients
        """
        request_type = request_tag // self.comm_size
        if request_type == MPITask.GET_CONFIGURATION.value:
            return self._get_configuration(client_id, request)
        elif request_type == MPITask.GET_GLOBAL_MODEL.value:
            return self._get_global_model(client_id, request)
        elif request_type == MPITask.UPDATE_GLOBAL_MODEL.value:
            return self._update_global_model(client_id, request)
        elif request_type == MPITask.INVOKE_CUSTOM_ACTION.value:
            return self._invoke_custom_action(client_id, request)
        else:
            raise ValueError(f"Invalid request tag: {request_tag}")
        
    def _get_configuration(
        self, 
        client_id: int, 
        request: MPITaskRequest
    ) -> MPITaskResponse:
        """
        Client requests the FL configurations that are shared among all clients from the server.
        :param: `client_id`: A unique client ID (only for logging purpose now)
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.meta_data`: JSON serialized FL configurations
        """
        self.logger.info(f"Received GetConfiguration request from client {client_id}")
        meta_data = json.loads(request.meta_data) if len(request.meta_data) > 0 else {}
        client_configs = self.server_agent.get_client_configs(**meta_data)
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        client_configs_serialized = json.dumps(client_configs)
        response = MPITaskResponse(
            status=MPIServerStatus.RUN.value,
            meta_data=client_configs_serialized,
        )
        return response
    
    def _get_global_model(
        self, 
        client_id: int, 
        request: MPITaskRequest
    ) -> Optional[MPITaskResponse]:
        """
        Return the global model to clients. This method is supposed to be called by clients to get the initial and final global model.
        :param: `client_id`: A unique client ID, which is the rank of the client in MPI (only for logging purpose now)
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.payload`: Serialized global model
        :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
        """
        self.logger.info(f"Received GetGlobalModel request from client {client_id}")
        meta_data = json.loads(request.meta_data) if len(request.meta_data) > 0 else {}
        model = self.server_agent.get_parameters(**meta_data, blocking=False)
        if not isinstance(model, Future):
            if isinstance(model, tuple):
                model = model[0]
                meta_data = json.dumps(model[1])
            else:
                meta_data = json.dumps({})
            model_serialized = model_to_byte(model)
            return MPITaskResponse(
                status=MPIServerStatus.RUN.value,
                payload=model_serialized,
                meta_data=meta_data,
            )
        else:
            self._global_model_futures[client_id] = model
            self._check_global_model_futures()
            return None

    def _update_global_model(
        self, 
        client_id: int, 
        request: MPITaskRequest
    ) -> Optional[MPITaskResponse]:
        """
        Update the global model with the local model from the client, 
        and return the updated global model to the client.
        :param: `client_id`: A unique client ID, which is the rank of the client in MPI.
        :param: `request.payload`: Serialized local model
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.payload`: Serialized updated global model
        :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
        """
        self.logger.info(f"Received UpdateGlobalModel request from client {client_id}")
        local_model = request.payload
        meta_data = json.loads(request.meta_data) if len(request.meta_data) > 0 else {}
        global_model = self.server_agent.global_update(client_id, local_model, blocking=False, **meta_data)
        if not isinstance(global_model, Future):
            if isinstance(global_model, tuple):
                meta_data = json.dumps(global_model[1])
                global_model = global_model[0]
            else:
                meta_data = json.dumps({})
            global_model_serialized = model_to_byte(global_model)
            status = MPIServerStatus.DONE.value if self.server_agent.training_finished() else MPIServerStatus.RUN.value
            return MPITaskResponse(
                status=status,
                payload=global_model_serialized,
                meta_data=meta_data,
            )
        else:
            self._global_model_futures[client_id] = global_model
            self._check_global_model_futures()
            return None

    def _invoke_custom_action(
        self,
        client_id: int,
        request: MPITaskRequest,
    ) -> Optional[MPITaskResponse]:
        """
        Invoke custom action on the server.
        :param: `client_id`: A unique client ID, which is the rank of the client in MPI (only for logging purpose now)
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
        """
        meta_data = json.loads(request.meta_data) if len(request.meta_data) > 0 else {}
        assert "action" in meta_data, "The action is not specified in the metadata"
        self.logger.info(f"Received InvokeCustomAction: {meta_data['action']} request from client {client_id}")
        action = meta_data["action"]
        del meta_data["action"]
        if action == "set_sample_size":
            meta_data["blocking"] = False
            ret_val = self.server_agent.set_sample_size(client_id, **meta_data)
            if ret_val is None:
                return MPITaskResponse(status=MPIServerStatus.RUN.value)
            else:
                self._meta_data_futures[client_id] = ret_val
                self._check_meta_data_futures()
                return None
        elif action == "close_connection":
            self.server_agent.close_connection(client_id)
            return MPITaskResponse(status=MPIServerStatus.DONE.value)
        elif action == "get_data_readiness_report":
            num_clients = self.server_agent.get_num_clients()
            if not hasattr(self, "_dr_metrics_lock"):
                self._dr_metrics = {}
                self._dr_metrics_client_ids = set()
                self._dr_metrics_lock = threading.Lock()                
            with self._dr_metrics_lock:
                self._dr_metrics_client_ids.add(client_id)
                for k, v in meta_data.items():
                    if k not in self._dr_metrics:
                        self._dr_metrics[k] = {}
                    self._dr_metrics[k][client_id] = v
                if len(self._dr_metrics_client_ids) == num_clients:
                    self.server_agent.data_readiness_report(self._dr_metrics)
                    response = MPITaskResponse(
                        status=MPIServerStatus.RUN.value,
                    )
                    response_bytes = response_to_byte(response)
                    for client_id in self._dr_metrics_client_ids:
                        self.comm.Send(response_bytes, dest=client_id, tag=client_id)
                    self._dr_metrics = {}
                    self._dr_metrics_client_ids = set()
            return None
        else:
            raise NotImplementedError(f"Custom action {action} is not implemented.")
        
    def _check_meta_data_futures(self):
        """
        Return the updated metadata to the client if the metadata `Future` object is available.
        """
        delete_keys = []
        for client_id, future in self._meta_data_futures.items():
            if future.done():
                meta_data = future.result()
                response = MPITaskResponse(
                    status=MPIServerStatus.RUN.value,
                    meta_data=json.dumps(meta_data),
                )
                response_bytes = response_to_byte(response)
                self.comm.Send(response_bytes, dest=client_id, tag=client_id)
                delete_keys.append(client_id)
        for key in delete_keys:
            del self._meta_data_futures[key]

    def _check_global_model_futures(self):
        """
        Return the updated global model to the client if the global model `Future` object is available.
        """
        delete_keys = []
        for client_id, future in self._global_model_futures.items():
            if future.done():
                global_model = future.result()
                if isinstance(global_model, tuple):
                    meta_data = json.dumps(global_model[1])
                    global_model = global_model[0]
                else:
                    meta_data = json.dumps({})
                global_model_serialized = model_to_byte(global_model)
                status = MPIServerStatus.DONE.value if self.server_agent.training_finished() else MPIServerStatus.RUN.value
                response = MPITaskResponse(
                    status=status,
                    payload=global_model_serialized,
                    meta_data=meta_data,
                )
                response_bytes = response_to_byte(response)
                self.comm.Send(response_bytes, dest=client_id, tag=client_id)
                delete_keys.append(client_id)
        for key in delete_keys:
            del self._global_model_futures[key]

    def _default_logger(self):
        """Create a default logger for the gRPC server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger