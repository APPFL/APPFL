"""
Running the ICEADMM algorithm using gRPC for FL. This example mainly shows 
the extendibility of the framework to support custom algorithms. In this case,
the server and clients need to communicate primal and dual states, and a  
penalty parameter. In addition, the clients also need to know its relative
sample size for local training purposes.
"""
from omegaconf import OmegaConf
from appfl.agent import APPFLClientAgent
from appfl.comm.grpc import GRPCClientCommunicator

client_agent_config = OmegaConf.load("config/client_1.yaml")

client_agent = APPFLClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id = client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_weight = client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size, sync=True)
client_agent.trainer.set_weight(client_weight["client_weight"])

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    new_global_model, metadata = client_communicator.update_global_model(local_model)
    if metadata['status'] == 'DONE':
        break
    if 'local_steps' in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
    client_agent.load_parameters(new_global_model)