"""
APPFL Hybrid Federated Learning Runner.

Demonstrates a *hybrid* server-driven federation in which different clients are
reached through different transports in a single run (here: Globus Compute + TES).
Each client declares its transport via ``comm_configs.comm_type`` in the client
config (or it is auto-inferred). The server code below is transport-agnostic --
it is identical to the Globus Compute / TES runners, only the communicator class
changes to ``HybridServerCommunicator``.
"""

import pprint
import argparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.comm.hybrid import HybridServerCommunicator

argparser = argparse.ArgumentParser(
    description="Hybrid (multi-transport) Federated Learning"
)
argparser.add_argument(
    "--server_config",
    type=str,
    default="./resources/config_hybrid/mnist/server.yaml",
    help="Path to server configuration file",
)
argparser.add_argument(
    "--client_config",
    type=str,
    default="./resources/config_hybrid/mnist/clients.yaml",
    help="Path to client configuration file",
)
argparser.add_argument(
    "--compute_token", required=False, help="Globus Compute authentication token"
)
argparser.add_argument(
    "--openid_token", required=False, help="Globus OpenID authentication token"
)
argparser.add_argument("--get_sample_size", action="store_true")
args = argparser.parse_args()

# Load server and client agent configurations
server_agent_config = OmegaConf.load(args.server_config)
client_agent_configs = OmegaConf.load(args.client_config)

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)
server_agent.logger.info(
    f"[Server] Will run for {server_agent_config.server_configs.num_global_epochs} global rounds"
)

# Create the hybrid server communicator. Transport-specific credentials (e.g.,
# Globus Compute tokens) are forwarded to every backend; each backend only
# consumes the keys it recognizes.
server_communicator = HybridServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs["clients"],
    logger=server_agent.logger,
    **(
        {
            "compute_token": args.compute_token,
            "openid_token": args.openid_token,
        }
        if args.compute_token is not None and args.openid_token is not None
        else {}
    ),
)

# Get sample size from clients
if args.get_sample_size:
    server_agent.logger.info("[Clients] Requesting sample sizes from all clients...")
    server_communicator.send_task_to_all_clients(task_name="get_sample_size")
    sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
    for client_id, sample_size in sample_size_ret.items():
        server_agent.set_sample_size(client_id, sample_size["sample_size"])

# Train the model
server_agent.logger.info("Starting Hybrid Federated Learning Training")
init_model = server_agent.get_parameters(globus_compute_run=True)
if isinstance(init_model, tuple):
    init_model, metadata = init_model[0], init_model[1]
else:
    metadata = {}
server_communicator.send_task_to_all_clients(
    task_name="train",
    model=init_model,
    metadata=metadata,
    need_model_response=True,
)

model_futures = {}
client_rounds = {}
server_agent.logger.info("[Training] Waiting for client updates...")

while not server_agent.training_finished():
    client_id, client_model, client_metadata = (
        server_communicator.recv_result_from_one_client()
    )

    # Track client round
    if client_id not in client_rounds:
        client_rounds[client_id] = 0
    client_rounds[client_id] += 1

    server_agent.logger.info(
        f"Received model from client {client_id} (Round {client_rounds[client_id]}), "
        f"with metadata:\n{pprint.pformat(client_metadata)}"
    )
    global_model = server_agent.global_update(
        client_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        metadata["round"] = client_rounds[client_id]
        if not server_agent.training_finished():
            server_communicator.send_task_to_one_client(
                client_id,
                task_name="train",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
            server_agent.logger.info(
                f"[Server] Sent updated global model to {client_id}"
            )

    # Deal with the model futures
    del_keys = []
    for client_id in model_futures:
        if model_futures[client_id].done():
            global_model = model_futures[client_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            metadata["round"] = client_rounds[client_id]
            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_id,
                    task_name="train",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
                server_agent.logger.info(
                    f"[Server] Sent updated global model to {client_id}"
                )
            del_keys.append(client_id)
    for key in del_keys:
        model_futures.pop(key)

# Cleanup
server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()

server_agent.logger.info("Hybrid Federated Learning Training Completed!")
server_agent.logger.info(
    f"Results saved to: {server_agent_config.server_configs.logging_output_dirname}"
)
