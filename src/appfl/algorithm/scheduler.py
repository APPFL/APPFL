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

class Scheduler:
    def __init__(self, comm: MPI.Comm, server: Any, emax: int, num_clients: int, num_global_epochs: int, lr: int, logger: Logger):
        self.iter = 0
        self.lr = lr
        self.comm = comm 
        self.server = server
        self.logger = logger
        self.num_clients = num_clients 
        self.num_global_epochs = num_global_epochs
        self.comm_size = comm.Get_size()
        self.group_counter = 0
        self.EMAX = emax
        self.EMIN = max(math.floor(0.2 * self.EMAX), 1)
        self.WARM_UP_EPOCH = 1
        self.SPEED_MOMENTUM = 0.9
        self.LATEST_TIME_FACTOR = 1.01
        self.LR_DECAY = 0.975
        self.client_info = {}
        self.group_of_arrival = OrderedDict()
        self.EMAX_BOUND = math.floor(1.2 * self.EMAX)

    def warmup(self):
        """Send each client few-epoch training task to sense the client computation speed."""
        # global_model = self.server.model.state_dict()
        # gloabl_model_buffer = io.BytesIO()
        # torch.save(global_model, gloabl_model_buffer)
        # global_model_bytes = gloabl_model_buffer.getvalue()
        # for i in range(1, self.num_clients+1):
        #     self.comm.send((len(global_model_bytes), False, self.WARM_UP_EPOCH, self.lr), dest=i, tag=i) 
        # for i in range(1, self.num_clients+1):
        #     self.comm.Isend(np.frombuffer(global_model_bytes, dtype=np.byte), dest=i, tag=i+self.comm_size)
        self.start_time = time.time()

    def local_update(self, local_model_size: int, client_idx: int):
        """Schedule update when receive information from one client."""
        self._record_info(client_idx)
        local_model = self._recv_model(local_model_size, client_idx)
        self._update(local_model, client_idx)

    def _update(self, local_model: dict, client_idx: int):
        self.iter += 1
        self.validation_flag = False
        # Get the client group
        group_idx = -1
        for group in self.group_of_arrival:
            if client_idx in self.group_of_arrival[group]['clients']:
                group_idx = group
                break
        if group_idx == -1:
            self._single_update(local_model, client_idx)
        else:
            self._group_update(local_model, client_idx, group_idx)

    def _single_update(self, local_model: dict, client_idx: int):
        """Update the global model using the local model itself."""
        # Update the global model
        self.server.update(local_model, self.client_info[client_idx]['step'], client_idx)
        self.client_info[client_idx]['step'] = self.server.global_step
        # Assign the client to a group of arrival
        self._assign_group(client_idx)
        self.validation_flag = True
        if self.iter < self.num_global_epochs:
            self._send_model(client_idx)
        else:
            self.server.update_all()

    def _group_update(self, local_model: dict, client_idx: int, group_idx: int):
        curr_time = time.time() - self.start_time
        # Update directly if the client arrives late
        if curr_time >= self.group_of_arrival[group_idx]['latest_arrival_time']:
            self.group_of_arrival[group_idx]['clients'].remove(client_idx)
            if len(self.group_of_arrival[group_idx]['clients']) == 0:
                del self.group_of_arrival[group_idx]
            self._single_update(local_model, client_idx)
        # Add the new coming model to the buffer and wait until group timer event gets triggered
        else:
            self.group_of_arrival[group_idx]['clients'].remove(client_idx)
            self.group_of_arrival[group_idx]['arrived_clients'].append(client_idx)
            self.server.buffer(local_model, self.client_info[client_idx]['step'], client_idx, group_idx)
            if len(self.group_of_arrival[group_idx]['clients']) == 0:
                self._group_aggregation(group_idx)

    def _assign_group(self, client_idx: int):
        """Assign a group to the client or create a new group for it when no suitable one exists."""
        curr_time = time.time() - self.start_time
        # Create a new group if no group exists at all
        if len(self.group_of_arrival) == 0:
            self.group_of_arrival[self.group_counter] = {
                'clients': [client_idx],
                'arrived_clients': [],
                'expected_arrival_time': curr_time + self.EMAX * self.client_info[client_idx]['speed'],
                'latest_arrival_time': curr_time + self.EMAX * self.client_info[client_idx]['speed'] * self.LATEST_TIME_FACTOR
            }
            # Add a timer event
            timer = threading.Timer(self.group_of_arrival[self.group_counter]['latest_arrival_time']-curr_time, self._group_aggregation, args=(self.group_counter, ))
            timer.start()
            self.client_info[client_idx]['goa'] = self.group_counter
            self.client_info[client_idx]['epoch'] = self.EMAX
            self.client_info[client_idx]['start_time'] = curr_time
            self.logger.info(f"Client {client_idx} - Create GOA {self.group_counter} - Local epoch {self.EMAX}")
            self.group_counter += 1
        # Assign the client to a group or create one for it
        else:
            if not self._join_group(client_idx):
                self._create_group(client_idx)

    def _join_group(self, client_idx: int):
        curr_time = time.time() - self.start_time
        assigned_group = -1     # assigned group for the client 
        assigned_epoch = -1     # assigned local training epochs for the client
        for group in self.group_of_arrival:
            remaining_time = self.group_of_arrival[group]['expected_arrival_time'] - curr_time
            local_epoch = math.floor(remaining_time / self.client_info[client_idx]['speed'])
            if local_epoch < self.EMIN or local_epoch < assigned_epoch or local_epoch > self.EMAX_BOUND:
                continue
            else:
                assigned_epoch = local_epoch
                assigned_group = group
        if assigned_group != -1:
            self.client_info[client_idx]['goa'] = assigned_group
            self.client_info[client_idx]['epoch'] = assigned_epoch
            self.client_info[client_idx]['start_time'] = curr_time
            self.group_of_arrival[assigned_group]['clients'].append(client_idx)
            self.logger.info(f"Client {client_idx} - Join GOA {assigned_group} - Local epoch {assigned_epoch}")
            return True
        else:
            return False

    def _create_group(self, client_idx: int):
        curr_time = time.time() - self.start_time
        # Calculate the assigned epoch for the client
        assigned_epoch = -1
        for group in self.group_of_arrival:
            if curr_time < self.group_of_arrival[group]['latest_arrival_time']:
                # Find the client with the fastest speed
                fastest_speed = float('inf')
                group_clients = self.group_of_arrival[group]['clients'] + self.group_of_arrival[group]['arrived_clients']
                for client in group_clients:
                    fastest_speed = min(fastest_speed, self.client_info[client]['speed'])
                est_arrival_time = self.group_of_arrival[group]['latest_arrival_time'] + fastest_speed * self.EMAX
                local_epoch = math.floor((est_arrival_time-curr_time) / self.client_info[client_idx]['speed'])
                if local_epoch <= self.EMAX:
                    assigned_epoch = max(assigned_epoch, local_epoch)
        assigned_epoch = self.EMIN if assigned_epoch >= 0 and assigned_epoch < self.EMIN else assigned_epoch
        assigned_epoch = self.EMAX if assigned_epoch < 0 else assigned_epoch
        # Create a group for the client
        self.group_of_arrival[self.group_counter] = {
            'clients': [client_idx],
            'arrived_clients': [],
            'expected_arrival_time': curr_time + assigned_epoch * self.client_info[client_idx]['speed'],
            'latest_arrival_time': curr_time + assigned_epoch * self.client_info[client_idx]['speed'] * self.LATEST_TIME_FACTOR
        }
        # Add a timer event
        timer = threading.Timer(self.group_of_arrival[self.group_counter]['latest_arrival_time']-curr_time, self._group_aggregation, args=(self.group_counter, ))
        timer.start()
        self.client_info[client_idx]['goa'] = self.group_counter
        self.client_info[client_idx]['epoch'] = assigned_epoch
        self.client_info[client_idx]['start_time'] = curr_time
        self.logger.info(f"Client {client_idx} - Create GOA {self.group_counter} - Local epoch {assigned_epoch}")
        self.group_counter += 1
    
    def _record_info(self, client_idx:int):
        """Record/update the client information for the coming client."""
        curr_time         = time.time() - self.start_time
        local_start_time  = self.client_info[client_idx]['start_time'] if client_idx in self.client_info else 0
        local_update_time = curr_time - local_start_time
        local_epoch       = self.client_info[client_idx]['epoch'] if client_idx in self.client_info else self.WARM_UP_EPOCH
        local_speed       = local_update_time / local_epoch 
        if client_idx not in self.client_info:
            self.client_info[client_idx] = {'speed': local_speed, 'step': 0, 'total_epochs': self.WARM_UP_EPOCH}
        else:
            self.client_info[client_idx]['speed'] = \
                (1-self.SPEED_MOMENTUM)*self.client_info[client_idx]['speed'] + self.SPEED_MOMENTUM*local_speed

    def _recv_model(self, local_model_size: int, client_idx: int):
        local_model_bytes = np.empty(local_model_size, dtype=np.byte)
        self.comm.Recv(local_model_bytes, source=client_idx+1, tag=client_idx+1+self.comm_size)
        local_model_buffer = io.BytesIO(local_model_bytes.tobytes())
        return torch.load(local_model_buffer)

    def _send_model(self, client_idx: int):
        # Record total epochs and decay the learning rate
        self.client_info[client_idx]['total_epochs'] += self.client_info[client_idx]['epoch']
        client_lr = self.lr * (self.LR_DECAY) ** (math.floor(self.client_info[client_idx]['total_epochs']/self.EMAX))
        self.logger.info(f"Total number of epochs for client{client_idx} is {self.client_info[client_idx]['total_epochs']}")
        self.logger.info(f"Learning rate for client{client_idx} is {client_lr}")

        global_model = self.server.model.state_dict()
        # Convert the updated model to bytes
        gloabl_model_buffer = io.BytesIO()
        torch.save(global_model, gloabl_model_buffer)
        global_model_bytes = gloabl_model_buffer.getvalue()
        # Send (buffer size, finish flag) - INFO - to the client in a blocking way
        self.comm.send((len(global_model_bytes), False, self.client_info[client_idx]['epoch'], client_lr), dest=client_idx+1, tag=client_idx+1)
        # Send the buffered model - MODEL - to the client in a blocking way
        self.comm.Send(np.frombuffer(global_model_bytes, dtype=np.byte), dest=client_idx+1, tag=client_idx+1+self.comm_size)

    def _group_aggregation(self, group_idx: int):        
        if group_idx in self.group_of_arrival:
            """Aggregate all the local gradients from a certain group."""
            # TODO: Do we need to add some lock?
            self.validation_flag = True
            self.server.update_group(group_idx) 
            client_speed = []
            for client in self.group_of_arrival[group_idx]['arrived_clients']:
                self.client_info[client]['step'] = self.server.global_step
                client_speed.append((client, self.client_info[client]['speed']))
            # sort clients in reverse order of speed, and assign group to clients (TODO: Check this)
            sorted_client_speed = sorted(client_speed, key=lambda x:x[1], reverse=False)
            self.group_of_arrival[group_idx]['expected_arrival_time'] = 0
            self.group_of_arrival[group_idx]['latest_arrival_time'] = 0
            for client, _ in sorted_client_speed:
                self._assign_group(client)
            # delete the group is not waiting any client
            if len(self.group_of_arrival[group_idx]['clients']) == 0:
                del self.group_of_arrival[group_idx]
            # Send the model if required
            if self.iter < self.num_global_epochs:
                for client, _ in sorted_client_speed:
                    self._send_model(client)
            else:
                self.server.update_all()
