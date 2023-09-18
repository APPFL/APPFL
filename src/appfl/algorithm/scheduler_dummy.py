import io
import math
import time
import threading
import numpy as np
from mpi4py import MPI
from typing import Any
from .algorithm import *
from torch.optim import *
from logging import Logger

class SchedulerDummy:
    def __init__(self, comm: MPI.Comm, server: Any, local_steps: int, num_clients: int, num_global_epochs: int, lr: float, logger: Logger):
        self.iter = 0
        self.comm = comm 
        self.server = server
        self.logger = logger
        self.num_clients = num_clients 
        self.num_global_epochs = num_global_epochs
        self.local_steps = local_steps
        self.lr = lr
        self.comm_size = comm.Get_size()
        self.client_info = {i: {'step': 0} for i in range(num_clients)}

    def local_update(self, local_model_size: int, client_idx: int):
        """Schedule update when receive information from one client."""
        local_model = self._recv_model(local_model_size, client_idx)
        self._update(local_model, client_idx)

    def _update(self, local_model: dict, client_idx: int):
        """Update the global model using the local model itself."""
        self.iter += 1
        self.validation_flag = True
        # Update the global model
        self.server.model.to("cpu")
        self.server.update(local_model, self.client_info[client_idx]['step'], client_idx)
        self.validation_flag = True
        self.client_info[client_idx]['step'] = self.server.global_step
        if self.iter < self.num_global_epochs:
            self._send_model(client_idx)

    def _recv_model(self, local_model_size: int, client_idx: int):
        local_model_bytes = np.empty(local_model_size, dtype=np.byte)
        self.comm.Recv(local_model_bytes, source=client_idx+1, tag=client_idx+1+self.comm_size)
        local_model_buffer = io.BytesIO(local_model_bytes.tobytes())
        return torch.load(local_model_buffer)

    def _send_model(self, client_idx: int):
        global_model = self.server.model.state_dict()
        # Convert the updated model to bytes
        gloabl_model_buffer = io.BytesIO()
        torch.save(global_model, gloabl_model_buffer)
        global_model_bytes = gloabl_model_buffer.getvalue()
        # Send (buffer size, finish flag) - INFO - to the client in a blocking way
        self.comm.send((len(global_model_bytes), False, self.local_steps, self.lr), dest=client_idx+1, tag=client_idx+1)
        # Send the buffered model - MODEL - to the client in a blocking way
        self.comm.Send(np.frombuffer(global_model_bytes, dtype=np.byte), dest=client_idx+1, tag=client_idx+1+self.comm_size)

