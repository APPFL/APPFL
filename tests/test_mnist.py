import os
import pytest
import torchvision
from mpi4py import MPI
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

# Prepare the MNIST data first for the tests
def readyMNISTdata():    
    currentpath = os.getcwd()    
    datafolderpath = os.path.join(currentpath, "_data")
    
    if not (os.path.exists(datafolderpath) and os.path.isdir(datafolderpath)):
        os.mkdir(datafolderpath)
    mnistfolderpath = os.path.join(datafolderpath, "MNIST")
    if not (os.path.exists(mnistfolderpath) and os.path.isdir(mnistfolderpath)):        
        print("Download MNIST data")
        torchvision.datasets.MNIST(
            "./_data", download=True, train=False, transform=ToTensor()
        )

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
if comm_size > 1:
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        readyMNISTdata()
    comm.Barrier()
else:
    # Serial
    readyMNISTdata()   

# Test for MPI Communication for FedAvg
@pytest.mark.mpi(min_size=2)
def test_mpi_fedavg():
    server_config = './tests/resources/configs/server_fedavg.yaml'
    client_config = './tests/resources/configs/client_1.yaml'
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
        server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
        if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
            server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
        # Create the server agent and communicator
        server_agent = ServerAgent(server_agent_config=server_agent_config)
        server_communicator = MPIServerCommunicator(comm, server_agent, logger=server_agent.logger)
        # Start the server to serve the clients
        server_communicator.serve()
    else:
        # Set the client configurations
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.train_configs.logging_id = f'Client{rank}'
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
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
        client_communicator.invoke_custom_action(action='close_connection')
            
            
# Test for MPI Communication for FedCompass
@pytest.mark.mpi(min_size=2)
def test_mpi_fedcompass():
    server_config = './tests/resources/configs/server_fedcompass.yaml'
    client_config = './tests/resources/configs/client_1.yaml'
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
        server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
        if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
            server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
        # Create the server agent and communicator
        server_agent = ServerAgent(server_agent_config=server_agent_config)
        server_communicator = MPIServerCommunicator(comm, server_agent, logger=server_agent.logger)
        # Start the server to serve the clients
        server_communicator.serve()
    else:
        # Set the client configurations
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.train_configs.logging_id = f'Client{rank}'
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
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
        client_communicator.invoke_custom_action(action='close_connection')
            
# Test for MPI Communication for FedAsync
@pytest.mark.mpi(min_size=2)
def test_mpi_fedasync():
    server_config = './tests/resources/configs/server_fedasync.yaml'
    client_config = './tests/resources/configs/client_1.yaml'
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
        server_agent_config.server_configs.scheduler_kwargs.num_clients = num_clients
        if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
            server_agent_config.server_configs.aggregator_kwargs.num_clients = num_clients
        # Create the server agent and communicator
        server_agent = ServerAgent(server_agent_config=server_agent_config)
        server_communicator = MPIServerCommunicator(comm, server_agent, logger=server_agent.logger)
        # Start the server to serve the clients
        server_communicator.serve()
    else:
        # Set the client configurations
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.train_configs.logging_id = f'Client{rank}'
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
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
        client_communicator.invoke_custom_action(action='close_connection')

# mpirun -n 2 python -m pytest --with-mpi