
import os
from collections import OrderedDict
from torch.optim import *
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
from mpi4py import MPI

import hydra
from omegaconf import DictConfig

import copy
import time
from algorithm.iadmm import *
from algorithm.fedavg import *
from models.cnn1 import *
from models.cnn2 import *
from read.coronahack import *
from read.femnist import *

def run_server(cfg: DictConfig, comm):

    ## Print and Write Results  
    dir = "../../../results"    
    filename = "Result_%s_%s_%s"%(cfg.dataset.type, cfg.model.type, cfg.fed.type)    
    if cfg.fed.type == "iadmm":  
        filename = "Result_%s_%s_%s(rho=%s)"%(cfg.dataset.type, cfg.model.type, cfg.fed.type, cfg.fed.args.penalty)
    
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

    ## Start    
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_clients = cfg.num_clients

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"
    
    model = eval(cfg.model.classname)(**cfg.dataset.size)
     
    if cfg.validation == True:
        if cfg.dataset.distributed == False:
            if cfg.dataset.torchvision == True:
                test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                    f"../../../datasets",
                    **cfg.dataset.args,
                    train=False,
                    transform=ToTensor(),                         
                )                        
            else:       
                test_data = eval(cfg.dataset.test)(**cfg.dataset.size)

            dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)
        else:
            test_data = eval(cfg.dataset.test)(**cfg.dataset.size)
            num_clients = test_data.num_clients
            dataloader = test_data.dataloader
 
    else:
        dataloader = None 

    # TODO: do we want to use root as a client?
    server = eval(cfg.fed.servername)(
        copy.deepcopy(model), 
        num_clients, 
        device, 
        dataloader=dataloader, 
        **cfg.fed.args
    ) 

    do_continue = True
    local_states = OrderedDict()    
    start_time = time.time()
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
            test_loss, accuracy = server.validation()
            log.info(
                f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
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
        print(title, end="")
        print(results, end="")
        outfile.write(results)

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)
 
    outfile.write("Device=%s \n"%(cfg.device))
    outfile.write("#Nodes=%s \n"%(comm_size))
    outfile.write("Instance=%s \n"%(cfg.dataset.type))
    outfile.write("#Clients=%s \n"%(num_clients))    
    outfile.write("Model=%s \n"%(cfg.model.type))
    outfile.write("Algorithm=%s \n"%(cfg.fed.type))
    outfile.write("Comm_Rounds=%s \n"%(cfg.num_epochs))
    outfile.write("Local_Epochs=%s \n"%(cfg.fed.args.num_local_epochs))    
    if cfg.fed.type == "iadmm":
        outfile.write("ADMM Penalty=%s \n"%(cfg.fed.args.penalty))

    outfile.close()


def run_client(cfg: DictConfig, comm):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_clients = cfg.num_clients
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)            
     
    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    model = eval(cfg.model.classname)(**cfg.dataset.size)    
    optimizer = eval(cfg.optim.classname)         

    dataloaders = []
    if cfg.dataset.distributed == False:             
        if cfg.dataset.torchvision == True:
            train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                f"../../../datasets",
                **cfg.dataset.args,
                train=True,
                transform=ToTensor(),            
            )        
        else:        
            train_data = eval(cfg.dataset.train)(**cfg.dataset.size)

        ## TO DO: advance techniques (e.g., utilizing batch)
        if cfg.fed.type == "iadmm":  
            cfg.batch_size = len(train_data)     

        for cid in num_client_groups[comm_rank - 1]:
            dataloaders.append(
                DataLoader(
                    train_data,
                    batch_size=cfg.batch_size,   
                    shuffle=False,                  
                    sampler=DistributedSampler(
                        train_data, num_replicas=num_clients, rank=cid
                    )
                )
            )
               
    else:
        train_data = eval(cfg.dataset.train)(**cfg.dataset.size)        

        num_clients = train_data.num_clients
        num_client_groups = np.array_split(range(num_clients), comm_size - 1)        
        
        for i, cid in enumerate(num_client_groups[comm_rank - 1]):  
            dataloaders.append([train_data.dataloader[cid]])
        
    
    clients = [
        eval(cfg.fed.clientname)(
            cid,
            copy.deepcopy(model),            
            optimizer,
            cfg.optim.args,
            dataloaders[i],
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


def run_serial(cfg: DictConfig):

    num_clients = cfg.num_clients
    num_epochs = cfg.num_epochs

    if cfg.dataset.torchvision == True:
        train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            "../../../datasets", **cfg.dataset.args, train=True, transform=ToTensor()
        )
        local_data_size = int(len(train_data) / num_clients)
        how_to_split = [local_data_size for i in range(num_clients)]
        how_to_split[-1] += len(train_data) - sum(how_to_split)
        datasets = data.random_split(train_data, how_to_split)
    else:
        raise NotImplementedError

    # print(cfg.model.classname)
    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    if cfg.validation == True:
        if cfg.dataset.torchvision == True:
            test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                "./datasets", **cfg.dataset.args, train=False, transform=ToTensor()
            )
        else:
            raise NotImplementedError

        server_dataloader = DataLoader(
            test_data, num_workers=0, batch_size=cfg.batch_size
        )
    else:
        server_dataloader = None

    server = eval(cfg.fed.servername)(
        model, num_clients, cfg.device, dataloader=server_dataloader
    )
    clients = [
        eval(cfg.fed.clientname)(
            k,
            model,
            optimizer,
            cfg.optim.args,
            DataLoader(
                datasets[k], num_workers=0, batch_size=cfg.batch_size, shuffle=True
            ),
            cfg.device,
            **cfg.fed.args,
        )
        for k in range(num_clients)
    ]
    local_states = OrderedDict()

    for t in range(num_epochs):
        global_state = server.model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_state)

        for k, client in enumerate(clients):
            client.model.load_state_dict(global_state)
            client.update()
            local_states[k] = client.model.state_dict()

        server.update(local_states)
        if cfg.validation == True:
            test_loss, accuracy = server.validation()
            log.info(
                f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
 
    torch.manual_seed(1)
 
    if comm_size > 1:
        if comm_rank == 0:
            run_server(cfg, comm)
        else:
            run_client(cfg, comm)
    else:
        run_serial(cfg)
    
    print("------DONE------", comm_rank)
  
if __name__ == "__main__":
    main()

# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./run.py

# mpiexec -np 5 python ./run.py
