import json
import pickle
import argparse
import time
from typing import Optional, Dict, Any, Tuple

from appfl.agent import ClientAgent
from appfl.config import ClientAgentConfig


class TESClientCommunicator:
    """
    GA4GH TES client-side communicator for APPFL.
    
    This class handles the client-side execution within TES containers,
    following the same patterns as other APPFL client communicators.
    """
    
    def __init__(self, client_agent: ClientAgent):
        self.client_agent = client_agent
    
    def load_model_from_path(self, model_path: str) -> Any:
        """Load serialized model from file path."""
        if not model_path:
            return None
        
        try:
            # The model data might be hex-encoded and gzip compressed
            import gzip
            import pickle
            
            with open(model_path, 'rb') as f:
                file_data = f.read()
                
            # Try to decompress directly first
            try:
                decompressed_data = gzip.decompress(file_data)
                model_data = pickle.loads(decompressed_data)
                return model_data
            except gzip.BadGzipFile:
                # If that fails, it might be hex-encoded, so decode first
                try:
                    hex_decoded = bytes.fromhex(file_data.decode('utf-8'))
                    decompressed_data = gzip.decompress(hex_decoded)
                    model_data = pickle.loads(decompressed_data)
                    return model_data
                except:
                    # If both fail, try loading as regular pickle
                    model_data = pickle.loads(file_data)
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
        
        This method follows the same task execution patterns as Ray and Globus Compute
        client communicators in APPFL.
        
        Args:
            task_name: Name of the task to execute ('train', 'evaluate', etc.)
            model_path: Path to input model file
            metadata_path: Path to metadata JSON file
            output_path: Path for output model
            logs_path: Path for training logs
            
        Returns:
            Tuple of (output_path, logs_path) for the results
        """
        try:
            import time
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
                
                # Save trained model
                if trained_model is not None:
                    self.save_model_to_path(trained_model, output_path)
                
                # Prepare training logs
                logs_data = {
                    "task_name": task_name,
                    "client_id": getattr(self.client_agent.client_agent_config, 'client_id', 'unknown'),
                    "training_logs": training_logs,
                    "timestamp": str(time.time()),
                    "metadata": metadata
                }
                
            elif task_name == "evaluate":
                # Evaluation task
                eval_result = self.client_agent.evaluate()
                
                # No model update for evaluation
                self.save_model_to_path(None, output_path)
                
                # Prepare evaluation logs
                logs_data = {
                    "task_name": task_name,
                    "client_id": getattr(self.client_agent.client_agent_config, 'client_id', 'unknown'),
                    "evaluation_result": eval_result,
                    "timestamp": str(time.time()),
                    "metadata": metadata
                }
                
            elif task_name == "get_sample_size":
                # Get dataset size
                sample_size = self.client_agent.get_sample_size()
                
                # No model update
                self.save_model_to_path(None, output_path)
                
                # Prepare sample size logs
                logs_data = {
                    "task_name": task_name,
                    "client_id": getattr(self.client_agent.client_agent_config, 'client_id', 'unknown'),
                    "sample_size": sample_size,
                    "timestamp": str(time.time()),
                    "metadata": metadata
                }
                print(f'DEBUG: Logs data for get_sample_size: {logs_data}')
            elif task_name == "get_parameters":
                # Get current model parameters
                current_params = self.client_agent.get_parameters()
                
                # Save current parameters
                self.save_model_to_path(current_params, output_path)
                
                # Prepare parameters logs
                logs_data = {
                    "task_name": task_name,
                    "client_id": getattr(self.client_agent.client_agent_config, 'client_id', 'unknown'),
                    "parameters_size": len(str(current_params)) if current_params else 0,
                    "timestamp": str(time.time()),
                    "metadata": metadata
                }
                
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            
            # Save logs
            self.save_logs_to_path(logs_data, logs_path)
            
            return output_path, logs_path
            
        except Exception as e:
            import time
            # Save error information
            error_logs = {
                "task_name": task_name,
                "client_id": getattr(self.client_agent.client_agent_config, 'client_id', 'unknown'),
                "status": "error",
                "timestamp": str(time.time()),
                "error": str(e),
                "metadata": metadata if 'metadata' in locals() else {}
            }
            
            self.save_logs_to_path(error_logs, logs_path)
            raise


def main():
    """Main entry point for TES client execution."""
    import time
    
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
            # Use minimal default configuration matching server setup
            print(f"DEBUG: Using default config (no config path provided)")
            default_config = {
                "train_configs": {
                    "trainer": "VanillaTrainer",
                    "mode": "step",
                    "num_local_steps": 100,
                    "optim": "Adam",
                    "optim_args": {"lr": 0.001},
                    "loss_fn_path": "/app/resources/loss/simple_loss.py",
                    "loss_fn_name": "SimpleLoss",
                    "do_validation": True,
                    "do_pre_validation": True,
                    "metric_path": "/app/resources/metric/acc.py",
                    "metric_name": "accuracy",
                    "use_dp": False,
                    "train_batch_size": 64,
                    "val_batch_size": 64,
                    "train_data_shuffle": True,
                    "val_data_shuffle": False
                },
                "model_configs": {
                    "model_path": "/app/resources/model/simple_model.py",
                    "model_name": "TinyNet",
                    "model_kwargs": {
                        "input_size": 8,
                        "num_classes": 2
                    }
                },
                "data_configs": {
                    "dataset_path": "/app/resources/dataset/simple_dataset.py",
                    "dataset_name": "get_tiny_data",
                    "dataset_kwargs": {
                        "num_clients": 2,
                        "client_id": 0,
                        "num_samples": 100
                    }
                }
            }
            from omegaconf import OmegaConf
            config_omega = OmegaConf.create(default_config)
            print(f"DEBUG: Default config created with keys: {list(config_omega.keys())}")
            print(f"DEBUG: Data configs keys: {list(config_omega.data_configs.keys())}")
            client_config = ClientAgentConfig(**config_omega)
            
        # Set client ID
        client_config.client_id = args.client_id
        
        # Create client agent
        print(f'DEBUG: Client config: {client_config}')
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
        
        print(f"Task '{args.task_name}' completed successfully")
        print(f"Output model: {output_path}")
        print(f"Training logs: {logs_path}")
        return 0
        
    except Exception as e:
        print(f"Error executing TES client task: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())