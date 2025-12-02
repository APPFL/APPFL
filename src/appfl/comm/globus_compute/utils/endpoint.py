import uuid
import time
from enum import Enum
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import Future
from globus_compute_sdk import Executor, MPIFunction
from typing import Optional, Union, Dict, OrderedDict, Tuple
from appfl.comm.globus_compute.globus_compute_client_communicator import (
    globus_compute_client_entry_point,
    data_transfer,
)


class ClientEndpointStatus(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1
    RUNNING = 2


class GlobusComputeClientEndpoint:
    """
    Represents a Globus Compute client endpoint, which can be
    used to submit tasks to the Globus Compute client endpoint.
    """

    def __init__(
        self,
        client_id: str,
        client_endpoint_id: str,
        transfer_endpoint_id: str,
        client_config: DictConfig,
    ):
        """
        :param `client_id`: Client ID, must be unique.
        :param `client_endpoint_id`: The Globus Compute client endpoint id. (Different clients may have the same endpoint id.)
        :param `client_config`: The client configuration for local training and communication.
        """
        self.client_id = client_id
        self.client_endpoint_id = client_endpoint_id
        self.transfer_endpoint_id = transfer_endpoint_id
        self.client_config = client_config
        self._set_no_runing_task()

    @property
    def status(self):
        """
        Get the status of the globus compute client enpdoint,
        and update the status if the client task is finished.
        """
        if self._status == ClientEndpointStatus.RUNNING:
            if self.future.done():
                self._set_no_runing_task()
        return self._status

    def _set_no_runing_task(self):
        """Clear running task for the federated learning client."""
        self.future = None
        self._status = ClientEndpointStatus.AVAILABLE
        self.task_name = "N/A"
        self.executing_task_id = None

    def cancel_task(self):
        """Cancel the currently running task."""
        self._set_no_runing_task()

    def submit_task(
        self,
        gce: Executor,
        transfer_gce: Executor,
        task_name: str,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        meta_data: Optional[Dict] = None,
    ) -> Tuple[Optional[str], Optional[Future]]:
        """
        Submit a task to the client's Globus Compute endpoint.
        :param `gce`: Globus Compute executor for submitting tasks to the Globus Compute client endpoint
        :param `task_name`: The name of the task to be submitted.
        :param `model`: [Optional] The model to be used for the task.
        :param `meta_data`: [Optional] The metadata for the task.
        :return `executing_task_id`: The ID of the task being executed.
        :return `future`: The future object for the task being executed.
        """
        if self.status != ClientEndpointStatus.AVAILABLE:
            return None, None
        gce.endpoint_id = self.client_endpoint_id
        transfer_gce.endpoint_id = self.transfer_endpoint_id
        self.client_config.start_time = time.time()
        if task_name == "get_sample_size_ds" or task_name == "train_ds":
            gce.resource_specification = {
                'num_nodes': self.client_config.ds_configs.num_nodes,      
                'ranks_per_node': self.client_config.ds_configs.ranks_per_node, 
            }

            # transfer data to client using the transfer_gce
            future = transfer_gce.submit(
                data_transfer,
                self.client_config,
                model,
                meta_data,
            )
            
            client_config_path, model_path, meta_data_path = future.result()

            total_ranks = self.client_config.ds_configs.num_nodes * self.client_config.ds_configs.ranks_per_node
            ranks_per_node = self.client_config.ds_configs.ranks_per_node

            mpi_cmd = (f"python -c 'from appfl.comm.globus_compute.globus_compute_client_communicator import globus_compute_client_entry_point_ds; "
                       f"globus_compute_client_entry_point_ds(\"{task_name}\", \"{client_config_path}\", \"{model_path}\", \"{meta_data_path}\")'")
            
            if self.client_config.scheduler == "PBS":
                if self.client_config.client_id == "Aurora":
                    MPI_globus_compute_client_entry_point = MPIFunction(f"true; mpiexec -n {total_ranks} -ppn {ranks_per_node} --cpu-bind=$CPU_BIND " + mpi_cmd)
                else:
                    MPI_globus_compute_client_entry_point = MPIFunction(f"true; mpiexec -n {total_ranks} -ppn {ranks_per_node} " + mpi_cmd)
            elif self.client_config.scheduler == "SLURM":
                MPI_globus_compute_client_entry_point = MPIFunction(f"true; srun --ntasks-per-node={ranks_per_node} --ntasks={total_ranks} --gpu-bind=none -l -u " + mpi_cmd)
                
            self.future = gce.submit(
                MPI_globus_compute_client_entry_point
            )
        else:
            self.future = gce.submit(
                globus_compute_client_entry_point,
                task_name=task_name,
                client_agent_config=self.client_config,
                model=model,
                meta_data=meta_data,
            )
        self._status = ClientEndpointStatus.RUNNING
        self.task_name = task_name
        self.executing_task_id = str(uuid.uuid4())
        return self.executing_task_id, self.future
