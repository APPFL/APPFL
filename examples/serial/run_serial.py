"""
Serial simulation of Federated learning.
It should be noted that only synchronous FL can be simulated in this way.
"""
import argparse
from omegaconf import OmegaConf
from appfl.agent import APPFLClientAgent, APPFLServerAgent

argparser = argparse.ArgumentParser()
argparser.add_argument("--server_config", type=str, default="config/server_fedavg.yaml")
argparser.add_argument("--client_config", type=str, default="config/client_1.yaml")
argparser.add_argument("--num_clients", type=int, default=10)
args = argparser.parse_args()

# Load server agent configurations and set the number of clients
server_agent_config = OmegaConf.load(args.server_config)
server_agent_config.server_configs.scheduler_kwargs.num_clients = args.num_clients
if hasattr(server_agent_config.server_configs.aggregator_kwargs, "num_clients"):
    server_agent_config.server_configs.aggregator_kwargs.num_clients = args.num_clients

# Create server agent
server_agent = APPFLServerAgent(server_agent_config=server_agent_config)

# Load base client configurations and set corresponding fields for different clients
client_agent_configs = [OmegaConf.load(args.client_config) for _ in range(args.num_clients)]
for i in range(args.num_clients):
    client_agent_configs[i].train_configs.logging_id = f'Client{i+1}'
    client_agent_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
    client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
    client_agent_configs[i].data_configs.dataset_kwargs.visualization = True if i == 0 else False

# Load client agents
client_agents = [
    APPFLClientAgent(client_agent_config=client_agent_configs[i]) 
    for i in range(args.num_clients)
]

# Get additional client configurations from the server
client_config_from_server = server_agent.get_client_configs()
for client_agent in client_agents:
    client_agent.load_config(client_config_from_server)

# Load initial global model from the server
init_global_model = server_agent.get_parameters(serial_run=True)
for client_agent in client_agents:
    client_agent.load_parameters(init_global_model)

# [Optional] Set number of local data to the server
for i in range(args.num_clients):
    sample_size = client_agents[i].get_sample_size()
    server_agent.set_sample_size(
        client_id=client_agents[i].get_id(), 
        sample_size=sample_size
    )

while not server_agent.training_finished():
    new_global_models = []
    for client_agent in client_agents:
        # Client local training
        client_agent.train()
        local_model = client_agent.get_parameters()
        # "Send" local model to server and get a Future object for the new global model
        # The Future object will be resolved when the server receives local models from all clients
        new_global_model_future = server_agent.global_update(
            client_id=client_agent.get_id(), 
            local_model=local_model,
            blocking=False,
        )
        new_global_models.append(new_global_model_future)
    # Load the new global model from the server
    for client_agent, new_global_model_future in zip(client_agents, new_global_models):
        client_agent.load_parameters(new_global_model_future.result())
