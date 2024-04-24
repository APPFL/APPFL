"""
Running the ADMM-based algorithm using MPI for FL. This example mainly shows 
the extendibility of the framework to support custom algorithms. In this case,
the server and clients need to communicate primal and dual states, and a  
penalty parameter. In addition, the clients also need to know its relative
sample size for local training purposes.
mpiexec -n 6 python  mpi/run_mpi_admm.py --server_config config/server_iiadmm.yaml
mpiexec -n 6 python  mpi/run_mpi_admm.py --server_config config/server_iceadmm.yaml
"""

import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import APPFLClientAgent, APPFLServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

argparse = argparse.ArgumentParser()
argparse.add_argument("--server_config", type=str, default="config/server_iceadmm.yaml")
argparse.add_argument("--client_config", type=str, default="config/client_1.yaml")
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

if rank == 0:
    # Load and set the server configurations
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
    if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
        server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
    # Create the server agent and communicator
    server_agent = APPFLServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(comm, server_agent, logger=server_agent.logger)
    # Start the server to serve the clients
    server_communicator.serve()
else:
    # Set the client configurations
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.train_configs.logging_id = f'Client{rank}'
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
    # Create the client agent and communicator
    client_agent = APPFLClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(comm, server_rank=0)
    # Load the configurations and initial global model
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)
    # (Specific to ICEADMM and IIADMM) Send the sample size to the server and set the client weight
    sample_size = client_agent.get_sample_size()
    client_weight = client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size, sync=True)
    client_agent.trainer.set_weight(client_weight["client_weight"])
    # Local training and global model update iterations
    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        new_global_model, metadata = client_communicator.update_global_model(local_model)
        if metadata['status'] == 'DONE':
            break
        if 'local_steps' in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
        client_agent.load_parameters(new_global_model)