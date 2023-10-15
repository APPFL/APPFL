import io
import time
import torch
import numpy as np
from mpi4py import MPI

class MpiCommunicator:
    """A general MPI communicator for synchronous or asynchronous federated learning experiments
    on multiple MPI processes, where each process can represent ONLY ONE federated learning client."""
    def __init__(self, comm):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.dests = []
        self.recv_queue = []
        self.queue_status = [False for _ in range(self.comm_size-1)]

    def scatter(self, contents, source):
        '''Scattering the contents to all clients from the source.'''
        if source == self.comm_rank:
            assert len(contents) == self.comm_size, "The size of the contents is not equal to the number of clients in scatter!"
        return self.comm.scatter(contents, root=source)
    
    def gather(self, content, dest):
        '''Gathering contents from all clients to the destination.'''
        return self.comm.gather(content, root=dest)
    
    def broadcast_global_model(self, model=None, args=None):
        '''Broadcast the global model state dict and additional arguments from FL server to FL clients.'''
        self.dests = [i for i in range(self.comm_size) if i != self.comm_rank] if len(self.dests) == 0 else self.dests
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            for i in self.dests:
                self.comm.send((0, args), dest=i, tag=i)
        else:
            model_buffer = io.BytesIO()
            torch.save(model, model_buffer)
            model_bytes = model_buffer.getvalue()
            for i in self.dests:
                if args is None:
                    self.comm.send(len(model_bytes), dest=i, tag=i)
                else:
                    self.comm.send((len(model_bytes), args), dest=i, tag=i)
            for i in self.dests:
                self.comm.Send(np.frombuffer(model_bytes, dtype=np.byte), dest=i, tag=i+self.comm_size)
            self.recv_queue = [self.comm.irecv(source=i, tag=i) for i in self.dests]
            self.queue_status = [True for _ in range(self.comm_size-1)]

    def send_global_model_to_client(self, model=None, args=None, client_idx=-1):
        '''Send the global model to a certain client.'''
        assert client_idx >= 0 and client_idx < self.comm_size, "Please provide a correct client idx!"
        self.dests = [i for i in range(self.comm_size) if i != self.comm_rank] if len(self.dests) == 0 else self.dests
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            self.comm.send((0, args), dest=self.dests[client_idx], tag=self.dests[client_idx])
        else:
            model_buffer = io.BytesIO()
            torch.save(model, model_buffer)
            model_bytes = model_buffer.getvalue()
            if args is None:
                self.comm.send(len(model_bytes), dest=self.dests[client_idx], tag=self.dests[client_idx])
            else:
                self.comm.send((len(model_bytes), args), dest=self.dests[client_idx], tag=self.dests[client_idx])
            self.comm.Send(np.frombuffer(model_bytes, dtype=np.byte), dest=self.dests[client_idx], tag=self.dests[client_idx]+self.comm_size)
            self.queue_status[client_idx] = True
            queue_idx = sum(self.queue_status[:client_idx])
            self.recv_queue.insert(queue_idx, self.comm.irecv(source=self.dests[client_idx], tag=self.dests[client_idx]))

    def send_local_model_to_server(self, model, dest):
        model_buffer = io.BytesIO()
        torch.save(model, model_buffer)
        model_bytes = model_buffer.getvalue()
        self.comm.isend(len(model_bytes), dest=dest, tag=self.comm_rank)
        self.comm.Isend(np.frombuffer(model_bytes, dtype=np.byte), dest=dest, tag=self.comm_rank+self.comm_size)

    def recv_global_model_from_server(self, source):
        '''Client receives the global model state dict from the server (source).'''
        info = self.comm.recv(source=source, tag=self.comm_rank)
        if isinstance(info, tuple):
            model_size, args = info[0], info[1]
        else:
            model_size, args = info, None
        if model_size == 0:
            return None, args
        model_bytes = np.empty(model_size, dtype=np.byte)
        self.comm.Recv(model_bytes, source=source, tag=self.comm_rank+self.comm_size)
        model_buffer = io.BytesIO(model_bytes.tobytes())
        model = torch.load(model_buffer)
        return model if args is None else (model, args)

    def recv_local_model_from_client(self):
        '''Receive a single local model from the front of the receiving queue.'''
        while True:
            time.sleep(1.0)     #TODO: Sometimes error occurs without a short sleep due to race condition
            queue_idx, model_size = MPI.Request.waitany(self.recv_queue)
            if queue_idx != MPI.UNDEFINED:
                model_bytes = np.empty(model_size, dtype=np.byte)
                self.recv_queue.pop(queue_idx)
                for i in range(len(self.queue_status)):
                    if self.queue_status[i]:
                        if queue_idx == 0:
                            client_idx = i
                            break
                        else:
                            queue_idx -= 1
                self.comm.Recv(model_bytes, source=self.dests[client_idx], tag=self.dests[client_idx]+self.comm_size)
                model_buffer = io.BytesIO(model_bytes.tobytes())
                model = torch.load(model_buffer)
                self.queue_status[client_idx] = False
                return client_idx, model
            
    def cleanup(self):
        for req in self.recv_queue:
            req.cancel()
