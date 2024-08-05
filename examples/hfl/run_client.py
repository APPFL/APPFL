import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from concurrent.futures import ThreadPoolExecutor
from appfl.communicator.grpc import GRPCClientCommunicator

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="./configs/hfl/client_1.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = ClientAgent(client_agent_config=client_agent_config)

# We create a list of client communicators to connect to multiple servers/intermediate nodes
client_communicators = [
    GRPCClientCommunicator(
        client_id = client_agent.get_id(),
        **server_kwargs,
    ) for server_kwargs in client_agent_config.comm_configs.grpc_configs.connect 
] if 'connect' in client_agent_config.comm_configs.grpc_configs else [
    GRPCClientCommunicator(
        client_id = client_agent.get_id(),
        **client_agent_config.comm_configs.grpc_configs,
    )
]

# We use a ThreadPoolExecutor to submit tasks to multiple communicators in a non-blocking way
connect_executor = ThreadPoolExecutor(max_workers=len(client_communicators)) 

client_config = client_communicators[0].get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicators[0].get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

round = 1
while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    # Do the checkpint here: feel free to add any custom checkpoint logic
    if client_agent_config.train_configs.get("do_checkpoint", False):
        if round % client_agent_config.train_configs.get("checkpoint_interval", 1) == 0:
            client_agent.save_checkpoint() # You can give a custom path here
    else: 
        print(f"Round {round} completed.")
    # Submit update_global_model tasks via all connect communicators
    global_model_futures = []
    for client_communicator in client_communicators:
        global_model_futures.append(
            connect_executor.submit(
                client_communicator.update_global_model, 
                local_model
            )
        )
    # Only retrieve the first global model (we assume all global models are the same)
    new_global_model, metadata = global_model_futures[0].result()
    if metadata['status'] == 'DONE':
        break
    if 'local_steps' in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']    
    client_agent.load_parameters(new_global_model)
    # YOU CAN ALSO DO ANOTHER CHECKPOINT HERE
    round += 1
        
# Close the connection with the server/intermediate node
for client_communicator in client_communicators:
    client_communicator.invoke_custom_action(action='close_connection')
client_agent.clean_up()