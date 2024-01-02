import yaml 
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
logging.info(f"Using MPI to simulate decentralized federated learning among {comm_size} clients.")

# Load the client graph for decentralized federated learning from a yaml file
def load_graph_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_graph(config):
    graph_type = config.get('graph_type', 'undirected')
    graph = defaultdict(list)

    if graph_type == 'directed':
        for client, data in config.get('clients', {}).items():
            graph[client] = data.get('neighbors', [])
    elif graph_type == 'undirected':
        for edge in config.get('edges', []):
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

    return graph

## Test
logging.info("=========Test for graph loading===========")
config_path_directed = './configs/graph_directed.yaml'
config_path_undirected = './configs/graph_undirected.yaml'

config_directed = load_graph_config(config_path_directed)
config_undirected = load_graph_config(config_path_undirected)

graph_directed = create_graph(config_directed)
graph_undirected = create_graph(config_undirected)
logging.info(f"Directed graph: {graph_directed}")
logging.info(f"Undirected graph: {graph_undirected}")
logging.info("==========================================")


# Prepare the datasets for decentralized federated learning - use the partitioned MNIST dataset
num_clients = len(graph_directed)
assert num_clients == comm_size, "The number of clients in the graph should be the same as the number of processes!"
logging.info("Loading the MNIST dataset using IID partitioning into {} client chunks.".format(num_clients))
import sys
sys.path.insert(0, '../')
from dataloader.mnist_dataloader import get_mnist

train_datasets, test_dataset = get_mnist(
    comm,
    num_clients=num_clients,
    partition="iid",
    visualization=True,
    output_dirname="./outputs"
)

# Prepare the model, loss function, and evaluation metric for decentralized federated learning - use a simple CNN model and cross-entropy loss
import torch
import torch.nn as nn
from models.cnn import CNN
from metric.acc import accuracy

logging.info("Using CNN, cross-entropy loss, and accuracy as the model, loss function, and evaluation metric, respectively.")
loss_fn = nn.CrossEntropyLoss(reduction="mean")
eval_metric = accuracy
torch.manual_seed(1) ## To ensure the same model initialization for all clients, we set the seed to 1
model = CNN(
    num_channel=1,
    num_classes=10,
    num_pixel=28
)

# Start training loops
from torch.utils.data import DataLoader
from appfl.config import Config
from appfl.algorithm.client_optimizer import ClientOptim

batch_size = 64
cfg = Config()
cfg.fed.args.num_local_epochs = 1

client_optim = ClientOptim(
    id = comm_rank,
    weight=None,
    model=model,
    loss_fn=loss_fn,
    dataloader=DataLoader(
        train_datasets[comm_rank],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    ),
    cfg=cfg,
    outfile=f"outputs/client_{comm_rank}.txt",
    test_dataloader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    ),
    metric=eval_metric,
    **cfg.fed.args,
)

# Map the client ID to the process rank
client_to_rank = {client: rank for rank, client in enumerate(sorted(graph_directed.keys()))}
rank_to_client = {rank: client for client, rank in client_to_rank.items()}
logging.info(client_to_rank)

from appfl.comm.mpi import MpiCommunicator
communicator = MpiCommunicator(comm)

global_epoch = 1
for epoch in range(global_epoch):
    logging.info(f"Epoch {epoch}")
    # local update
    state = client_optim.update()
    # send state to neighbors
    for neighbor in graph_directed[rank_to_client[comm_rank]]:
        logging.info(f"{rank_to_client[comm_rank]} Sending state to {neighbor}")
        communicator.send_local_model_to_server(state, dest=client_to_rank[neighbor])
    # receive state from neighbors
    client_states = [state]
    for client in graph_directed:
        if rank_to_client[comm_rank] in graph_directed[client]:
            logging.info(f"{rank_to_client[comm_rank]} Receiving state from {client}")
            state = communicator.recv_global_model_from_server(source=client_to_rank[client])
            logging.info(f"{rank_to_client[comm_rank]} Received state from {client}")
            client_states.append(state)
    # aggregate states
    for name, param in client_optim.model.named_parameters():
        param.data = torch.mean(torch.stack([state[name] for state in client_states]), dim=0)
        



