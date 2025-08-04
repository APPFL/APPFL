"""
APPFL TES Server Example

This example demonstrates how to run a federated learning server
using GA4GH Task Execution Service (TES) for client execution.
"""

import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.tes import TESServerCommunicator
from appfl.config import ServerAgentConfig, ClientAgentConfig


def main():
    parser = argparse.ArgumentParser(description="APPFL TES Server")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../resources/configs/mnist/server_tes.yaml",
        help="Path to server configuration file"
    )
    parser.add_argument(
        "--num-clients",
        type=int, 
        default=2,
        help="Number of clients"
    )
    
    args = parser.parse_args()
    
    # Load server configuration
    server_config_dict = OmegaConf.load(args.config)
    server_agent_config = ServerAgentConfig(**server_config_dict)
    
    # Create client configurations
    client_agent_configs = []
    for i in range(args.num_clients):
        client_config = ClientAgentConfig(**server_config_dict.client_configs)
        client_config.client_id = f"tes_client_{i+1}"
        client_agent_configs.append(client_config)
    
    # Create server agent
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    
    # Create TES communicator
    tes_communicator = TESServerCommunicator(
        server_agent_config=server_agent_config,
        client_agent_configs=client_agent_configs
    )
    
    print(f"Starting APPFL federated learning with TES")
    print(f"TES Endpoint: {tes_communicator.tes_endpoint}")
    print(f"Number of clients: {len(client_agent_configs)}")
    print(f"Docker image: {tes_communicator.docker_image}")
    
    try:
        # Run federated learning
        num_global_epochs = server_agent_config.server_configs.num_global_epochs
        
        for epoch in range(num_global_epochs):
            print(f"\n=== Global Epoch {epoch + 1}/{num_global_epochs} ===")
            
            # Get current global model
            global_model = server_agent.get_parameters()
            
            # Send training task to all clients
            print("Submitting training tasks to TES...")
            futures = tes_communicator.send_task_to_all_clients(
                task_name="train",
                model=global_model,
                metadata={"epoch": epoch + 1},
                need_model_response=True
            )
            
            print(f"Submitted {len(futures)} tasks to TES")
            print("Waiting for training completion...")
            
            # Receive results from all clients
            client_results, client_metadata = tes_communicator.recv_result_from_all_clients()
            
            print(f"Received results from {len(client_results)} clients")
            
            # Aggregate models
            for client_id, local_model in client_results.items():
                if local_model is not None:
                    print(f"Processing update from {client_id}")
                    server_agent.global_update(client_id, local_model, blocking=True)
                else:
                    print(f"Warning: No model received from {client_id}")
            
            # Log epoch completion
            print(f"Epoch {epoch + 1} completed")
            
            # Optional: Evaluate global model
            if hasattr(server_agent, 'evaluate'):
                eval_results = server_agent.evaluate()
                print(f"Global model evaluation: {eval_results}")
        
        print(f"\nFederated learning completed successfully!")
        
        # Get final global model
        final_model = server_agent.get_parameters()
        print("Final global model obtained")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during federated learning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Shutting down TES communicator...")
        tes_communicator.shutdown_all_clients()
        print("Shutdown complete")


if __name__ == "__main__":
    main()