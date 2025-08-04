import json
import pickle
import argparse
from typing import Optional, Dict, Any, Tuple

from appfl.agent import ClientAgent
from appfl.config import ClientAgentConfig


class TESClientCommunicator:
    """
    GA4GH TES client-side communicator for APPFL.
    
    This class handles the client-side execution within TES containers,
    including loading models, running training, and preparing outputs.
    """
    
    def __init__(self, client_agent: ClientAgent):
        self.client_agent = client_agent
    
    def load_model_from_path(self, model_path: str) -> Any:
        """Load serialized model from file path."""
        if not model_path:
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def load_metadata_from_path(self, metadata_path: str) -> Dict:
        """Load metadata from JSON file path."""
        if not metadata_path:
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {metadata_path}: {e}")
    
    def save_model_to_path(self, model: Any, output_path: str):
        """Save model to file path."""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {output_path}: {e}")
    
    def save_logs_to_path(self, logs: Dict, logs_path: str):
        """Save training logs to JSON file path."""
        try:
            with open(logs_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save logs to {logs_path}: {e}")
    
    def execute_task(
        self, 
        task_name: str, 
        model_path: str = None, 
        metadata_path: str = None,
        output_path: str = "/tmp/output_model.pkl",
        logs_path: str = "/tmp/training_logs.json"
    ) -> Tuple[str, str]:
        """
        Execute a federated learning task within the TES container.
        
        Returns paths to the output model and logs files.
        """
        try:
            # Load inputs
            model_data = self.load_model_from_path(model_path)
            metadata = self.load_metadata_from_path(metadata_path)
            
            # Load model parameters if provided
            if model_data is not None:
                self.client_agent.load_parameters(model_data)
            
            # Execute the task based on task_name
            if task_name == "train":
                # Standard training task
                trained_model, training_logs = self.client_agent.train()
                
            elif task_name == "evaluate":
                # Evaluation task
                eval_result = self.client_agent.evaluate()
                trained_model = None  # No model update for evaluation
                training_logs = {"evaluation_result": eval_result}
                
            elif task_name == "get_sample_size":
                # Get dataset size
                sample_size = self.client_agent.get_sample_size()
                trained_model = None
                training_logs = {"sample_size": sample_size}
                
            elif task_name == "get_parameters":
                # Get current model parameters
                trained_model = self.client_agent.get_parameters()
                training_logs = {"message": "Model parameters retrieved"}
                
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            
            # Prepare logs with metadata
            output_logs = {
                "task_name": task_name,
                "client_id": self.client_agent.get_id(),
                "timestamp": str(time.time()),
                "training_logs": training_logs,
                "metadata": metadata
            }
            
            # Save outputs
            if trained_model is not None:
                self.save_model_to_path(trained_model, output_path)
            
            self.save_logs_to_path(output_logs, logs_path)
            
            return output_path, logs_path
            
        except Exception as e:
            # Save error logs
            error_logs = {
                "task_name": task_name,
                "client_id": self.client_agent.get_id(),
                "timestamp": str(time.time()),
                "error": str(e),
                "metadata": metadata if 'metadata' in locals() else {}
            }
            
            self.save_logs_to_path(error_logs, logs_path)
            raise


def main():
    """Main entry point for TES client execution."""
    parser = argparse.ArgumentParser(description="APPFL TES Client Runner")
    parser.add_argument("--task-name", required=True, help="Name of the task to execute")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--config-path", help="Path to client configuration file")
    parser.add_argument("--model-path", help="Path to input model file")
    parser.add_argument("--metadata-path", help="Path to metadata JSON file")
    parser.add_argument("--output-path", default="/tmp/output_model.pkl", 
                        help="Path for output model")
    parser.add_argument("--logs-path", default="/tmp/training_logs.json",
                        help="Path for training logs")
    
    args = parser.parse_args()
    
    try:
        # Create client agent
        if args.config_path:
            # Load configuration from file
            from omegaconf import OmegaConf
            config = OmegaConf.load(args.config_path)
            client_config = ClientAgentConfig(**config)
        else:
            # Use default configuration
            client_config = ClientAgentConfig()
            
        # Set client ID
        client_config.client_id = args.client_id
        
        # Create client agent
        client_agent = ClientAgent(client_config)
        
        # Create TES communicator
        tes_client = TESClientCommunicator(client_agent)
        
        # Execute the task
        output_path, logs_path = tes_client.execute_task(
            task_name=args.task_name,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            output_path=args.output_path,
            logs_path=args.logs_path
        )
        
        print(f"Task completed successfully.")
        print(f"Output model: {output_path}")
        print(f"Training logs: {logs_path}")
        
    except Exception as e:
        print(f"Task failed: {e}")
        exit(1)


if __name__ == "__main__":
    import time
    main()