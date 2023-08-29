import io
import torch
import numpy as np
from mpi4py import MPI

class MpiCommunicator:
    def __init__(self, comm):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.recv_queue = []
        self.dests = []

    def scatter(self, contents, source):
        '''Scattering the contents to all clients from the source.'''
        assert len(contents) == self.comm_size, "The size of the contents is not equal to the number of clients in scatter!"
        return self.comm.scatter(contents, root=source)
    
    def gather(self, content, dest):
        '''Gathering contents from all clients to the destination.'''
        return self.comm.gather(content, root=dest)
    
    def broadcast_global_model(self, model, args=None):
        '''Broadcast the global model state dict and additional arguments from FL server to FL clients.'''
        self.dests = [i for i in range(self.comm_size) if i != self.comm_rank] if len(self.dests) == 0 else self.dests
        model_buffer = io.BytesIO()
        torch.save(model, model_buffer)
        model_bytes = model_buffer.getvalue()
        for i in self.dests:
            if args is None:
                self.comm.send(len(model_bytes), dest=i, tag=i)
            else:
                self.comm.send((len(model_bytes, args)), dest=i, tag=i)
        for i in self.dests:
            self.comm.Isend(np.frombuffer(model_bytes, dtype=np.byte), dest=i, tag=i+self.comm_size)
        self.recv_queue = [self.comm.irecv(source=i, tag=i) for i in self.dests]

    def send_single_global_model(self, model, args=None, client_idx=-1):
        '''Send the global model to a certain client.'''
        assert client_idx >= 0 and client_idx < self.comm_size, "Please provide a correct client idx!"
        self.dests = [i for i in range(self.comm_size) if i != self.comm_rank] if len(self.dests) == 0 else self.dests
        model_buffer = io.BytesIO()
        torch.save(model, model_buffer)
        model_bytes = model_buffer.getvalue()
        if args is None:
            self.comm.send(len(model_bytes), dest=self.dests[client_idx], tag=self.dests[client_idx])
        else:
            self.comm.send((len(model_bytes), args), dest=self.dests[client_idx], tag=self.dests[client_idx])
        self.comm.Send(np.frombuffer(model_bytes, dtype=np.byte), dest=self.dests[client_idx], tag=self.dests[client_idx]+self.comm_size)
        self.recv_queue.insert(client_idx, self.comm.irecv(source=self.dests[client_idx], tag=self.dests[client_idx]))

    def recv_global_model(self, source):
        '''Client receives the global model state dict from the server (source).'''
        info = self.comm.recv(source=source, tag=self.comm_rank)
        if isinstance(info, tuple):
            model_size, args = info[0], info[1]
        else:
            model_size, args = info, None
        model_bytes = np.empty(model_size, dtype=np.byte)
        self.comm.Recv(model_bytes, source=source, tag=self.comm_rank+self.comm_size)
        model_buffer = io.BytesIO(model_bytes.tobytes())
        model = torch.load(model_buffer)
        return model if args is None else (model, args)

    def recv_single_local_model(self):
        '''Receive a single local model from the front of the receiving queue.'''
        while True:
            client_idx, model_size = MPI.Request.waitany(self.recv_queue)
            if client_idx != MPI.UNDEFINED:
                model_bytes = np.empty(model_size, dtype=np.byte)
                self.comm.Recv(model_bytes, source=self.dests[client_idx], tag=self.dests[client_idx]+self.comm_size)
                model_buffer = io.BytesIO(model_bytes.tobytes())
                model = torch.load(model_buffer)
                self.recv_queue.pop(client_idx)
                return client_idx, model
            
    def cleanup(self):
        for req in self.recv_queue:
            req.cancel()



        
        


