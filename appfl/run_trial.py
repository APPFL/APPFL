
import os

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *

from torch.utils.data import Dataset, DataLoader

import numpy as np

from omegaconf import DictConfig

import copy
import time
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


def run_serial(cfg: DictConfig, model: nn.Module, train_data: Dataset, test_data: Dataset):

    num_clients = len(train_data)
    num_epochs = cfg.num_epochs

    optimizer = eval(cfg.optim.classname)

    server_dataloader = DataLoader(test_data, num_workers=0, batch_size=cfg.batch_size, shuffle=False)

    server = eval(cfg.fed.servername)(
            copy.deepcopy(model), 
            num_clients, 
            cfg.device,
            **cfg.fed.args
        )

        
    clients = [
        eval(cfg.fed.clientname)(
            k,
            copy.deepcopy(model),
            optimizer,
            cfg.optim.args,
            DataLoader(
                train_data[k], num_workers=0, batch_size=cfg.batch_size, shuffle=False
            ),
            cfg.device,
            **cfg.fed.args,
        )
        for k in range(num_clients)
    ]
 

    local_states = OrderedDict()

    for t in range(num_epochs):
        global_state = server.model.state_dict()
        # for client in clients:
        #     client.model.load_state_dict(global_state)

        for k, client in enumerate(clients):            
            client.model.load_state_dict(global_state)
            client.update()
            local_states[k] = client.model.state_dict()

        server.update(global_state, local_states)
    
        test_loss, accuracy = validation(server, server_dataloader)
        log.info(
            f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

 
def run_server(cfg: DictConfig, comm, model: nn.Module, test_dataset: Dataset, num_clients: int, DataSet_name: str ):

    ## Print and Write Results  
    dir = "../../../results" 
    if os.path.isdir(dir) == False:
        os.mkdir(dir)            
    filename = "Result_%s_%s"%(DataSet_name, cfg.fed.type)    
    if cfg.fed.type == "iadmm":  
        filename = "Result_%s_%s(rho=%s)"%(DataSet_name, cfg.fed.type, cfg.fed.args.penalty)
    
    file_ext = ".txt"
    file = dir+"/%s%s"%(filename,file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir+"/%s_%d%s"%(filename, uniq, file_ext)
        uniq += 1
    outfile = open(file,"w")
    title = (
            "%12s %12s %12s %12s %12s %12s %12s \n"
            % (
                "Iter",                
                "Local[s]",
                "Global[s]",
                "Iter[s]",
                "Elapsed[s]",
                "TestAvgLoss",
                "TestAccuracy"                
            )
        )    
    outfile.write(title)
    print(title, end="")

    ## Start    
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    server_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=cfg.batch_size, shuffle=False)

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
            # log.info(
            #     f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            # )
        PerIter_time = time.time() - PerIter_start
        Elapsed_time = time.time() - start_time
        
        ##
        results = (
            "%12d %12.2f %12.2f %12.2f %12.2f %12.6f %12.2f \n"
            % (
                t+1,
                LocalUpdate_time,
                GlobalUpdate_time,
                PerIter_time,
                Elapsed_time,
                test_loss,
                accuracy                 
            )
        )        
        print(results, end="")
        outfile.write(results)

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)
 
    outfile.write("Device=%s \n"%(cfg.device))
    outfile.write("#Nodes=%s \n"%(comm_size))
    outfile.write("Dataset=%s \n"%(DataSet_name))
    outfile.write("#Clients=%s \n"%(num_clients))        
    outfile.write("Algorithm=%s \n"%(cfg.fed.type))
    outfile.write("Comm_Rounds=%s \n"%(cfg.num_epochs))
    outfile.write("Local_Epochs=%s \n"%(cfg.fed.args.num_local_epochs))    
    outfile.write("Elapsed_time=%s \n"%(round(Elapsed_time,2)))  
    outfile.write("BestAccuracy=%s \n"%(BestAccuracy))      
    
    if cfg.fed.type == "iadmm":
        outfile.write("ADMM Penalty=%s \n"%(cfg.fed.args.penalty))

    outfile.close()


def run_client(cfg: DictConfig, comm, model: nn.Module, clients_dataloaders: DataLoader, num_client_groups: list):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()    
    
     
    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device
       
    optimizer = eval(cfg.optim.classname)         
        
    clients = [
        eval(cfg.fed.clientname)(
            cid,
            copy.deepcopy(model),            
            optimizer,
            cfg.optim.args,
            clients_dataloaders[i],            
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

