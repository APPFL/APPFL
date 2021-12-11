
import os

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import Dataset, DataLoader

import numpy as np

from omegaconf import DictConfig

import copy
import time
from .utils import *
from .algorithm.iadmm import *
from .algorithm.fedavg import *

def validation(self, dataloader):
            
    if dataloader is not None:
        self.loss_fn = CrossEntropyLoss()
    else:
        self.loss_fn = None

    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    self.model.to(self.device)
    self.model.eval()
    test_loss = 0
    correct = 0
    tmpcnt=0; tmptotal=0
    with torch.no_grad():
        for img, target in dataloader:
            tmpcnt+=1; tmptotal+=len(target)
            img = img.to(self.device)
            target = target.to(self.device)
            logits = self.model(img) 
            test_loss += self.loss_fn(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # FIXME: do we need to sent the model to cpu again?
    # self.model.to("cpu")
    
    test_loss = test_loss / tmpcnt
    accuracy = 100.0 * correct / tmptotal

    return test_loss, accuracy


def run_serial(cfg: DictConfig, model: nn.Module, train_data: Dataset, test_data: Dataset, DataSet_name: str):

    outfile = print_write_result_title(cfg, DataSet_name)

    num_clients = len(train_data)
    num_epochs = cfg.num_epochs

    optimizer = eval(cfg.optim.classname)

    server_dataloader = DataLoader(
        test_data, 
        num_workers=0, 
        batch_size=cfg.test_data_batch_size, 
        shuffle=cfg.test_data_shuffle
        )

    server = eval(cfg.fed.servername)(
            copy.deepcopy(model), 
            num_clients, 
            cfg.device,
            **cfg.fed.args
        )
    
    batchsize={}  
    for k in range(num_clients):            
        batchsize[k] = cfg.train_data_batch_size
        if cfg.fed.type == "iadmm":        
            batchsize[k] = len(train_data[k])

        
    clients = [
        eval(cfg.fed.clientname)(
            k,
            copy.deepcopy(model),
            optimizer,
            cfg.optim.args,
            DataLoader(
                train_data[k], 
                num_workers=0, 
                batch_size=batchsize[k], 
                shuffle=cfg.train_data_shuffle
            ),
            cfg.device,
            **cfg.fed.args,
        )
        for k in range(num_clients)
    ]
 

    local_states = OrderedDict()

    start_time = time.time()
    BestAccuracy = 0.0    
    for t in range(num_epochs):
        PerIter_start = time.time()

        global_state = server.model.state_dict()
        LocalUpdate_start = time.time()     
        for k, client in enumerate(clients):            
            client.model.load_state_dict(global_state)
            client.update()
            local_states[k] = client.model.state_dict()
        LocalUpdate_time = time.time() - LocalUpdate_start

        GlobalUpdate_start = time.time()
        server.update(global_state, local_states)
        GlobalUpdate_time = time.time() - GlobalUpdate_start
        
        if cfg.validation == True:            
            test_loss, accuracy = validation(server, server_dataloader)

            if accuracy > BestAccuracy:
                BestAccuracy = accuracy    
        
        PerIter_time = time.time() - PerIter_start
        Elapsed_time = time.time() - start_time

        outfile = print_write_result_iteration(
                outfile,
                t,LocalUpdate_time, GlobalUpdate_time, 
                PerIter_time, Elapsed_time,
                test_loss,accuracy
                )   


    print_write_result_summary(cfg, outfile, 1, DataSet_name, num_clients, Elapsed_time, BestAccuracy)    
        
 
def run_server(cfg: DictConfig, comm, model: nn.Module, test_dataset: Dataset, num_clients: int, DataSet_name: str ):
    
    outfile = print_write_result_title(cfg, DataSet_name)

    ## Start    
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
 
    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    server_dataloader = DataLoader(
        test_dataset, 
        num_workers=0, 
        batch_size=cfg.test_data_batch_size, 
        shuffle=cfg.test_data_shuffle
        )

    # TODO: do we want to use root as a client?
    server = eval(cfg.fed.servername)(
        copy.deepcopy(model), 
        num_clients, 
        device,         
        **cfg.fed.args
    )     

    do_continue = True
    local_states = OrderedDict()    
    start_time = time.time()
    BestAccuracy = 0.0    
    for t in range(cfg.num_epochs):
        PerIter_start = time.time()
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")
        
        global_state = server.model.state_dict()        
        
        LocalUpdate_start = time.time()     
        global_state = comm.bcast(global_state, root=0)         
        gathered_states = comm.gather(None, root=0)
        LocalUpdate_time = time.time() - LocalUpdate_start

        GlobalUpdate_start = time.time()
        for i, states in enumerate(gathered_states):            
            if states is not None:
                for sid, state in states.items():                    
                    local_states[sid] = state                           
        server.update(global_state, local_states)
        GlobalUpdate_time = time.time() - GlobalUpdate_start

        if cfg.validation == True:            
            test_loss, accuracy = validation(server, server_dataloader)

            if accuracy > BestAccuracy:
                BestAccuracy = accuracy
\
        PerIter_time = time.time() - PerIter_start
        Elapsed_time = time.time() - start_time
        
        outfile = print_write_result_iteration(
                outfile,
                t,LocalUpdate_time, GlobalUpdate_time, 
                PerIter_time, Elapsed_time,
                test_loss,accuracy
                )         

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)
    
    print_write_result_summary(cfg, outfile, comm_size, DataSet_name, num_clients, Elapsed_time, BestAccuracy)
    

def run_client(cfg: DictConfig, comm, model: nn.Module, train_datasets: Dataset, num_clients: int):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()    

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device
       
    optimizer = eval(cfg.optim.classname)         

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)           

    batchsize={}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        batchsize[cid] = cfg.train_data_batch_size
        if cfg.fed.type == "iadmm":                      
            batchsize[cid] = len(train_datasets[cid])
        

    clients = [
        eval(cfg.fed.clientname)(
            cid,
            copy.deepcopy(model),            
            optimizer,
            cfg.optim.args,
            DataLoader( 
                train_datasets[cid], 
                num_workers=0, 
                batch_size=batchsize[cid], 
                shuffle=cfg.train_data_shuffle
            ),
            device,
            **cfg.fed.args,
        )
        for i, cid in enumerate(num_client_groups[comm_rank - 1])
    ] 
 
    do_continue = comm.bcast(None, root=0)
    local_states = OrderedDict()

    while do_continue:
        global_state = comm.bcast(None, root=0)
        
        # assign the globl state to the clients first (to avoid potential shallow copy)
        for client in clients:
            client.model.load_state_dict(global_state)            
            client.update()         
            
            # We need to load the model on cpu, before communicating.
            # Otherwise, out-of-memeory error from GPU
            client.model.to("cpu")
            local_states[client.id] = client.model.state_dict()
            
        comm.gather(local_states, root=0)
        do_continue = comm.bcast(None, root=0)

