import json
import uuid
import time
import requests
from typing import Dict, List, Optional, Union, OrderedDict, Tuple, Any
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from appfl.comm.base import BaseServerCommunicator
from appfl.comm.utils.config import ClientTask
from appfl.config import ClientAgentConfig, ServerAgentConfig
from appfl.logger import ServerAgentFileLogger


class TESServerCommunicator(BaseServerCommunicator):
    """
    GA4GH Task Execution Service (TES) server communicator for APPFL.
    
    This communicator enables APPFL to submit federated learning tasks to
    GA4GH TES-compliant compute infrastructures.
    """
    
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs
    ):
        self.comm_type = "tes"
        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)
        
        # TES-specific configuration
        self.tes_endpoint = self._get_tes_config("tes_endpoint")
        self.auth_token = self._get_tes_config("auth_token", required=False)
        self.docker_image = self._get_tes_config("docker_image", "appfl/client:latest")
        self.resource_requirements = self._get_tes_config("resource_requirements", {})
        
        # Default resource requirements
        default_resources = {
            "cpu_cores": 1,
            "ram_gb": 2.0,
            "disk_gb": 10.0,
            "preemptible": False
        }
        self.resource_requirements = {**default_resources, **self.resource_requirements}
        
        self.executor = ThreadPoolExecutor(max_workers=len(client_agent_configs))
        self.logger.info(f"TES Server Communicator initialized with endpoint: {self.tes_endpoint}")
    
    def _get_tes_config(self, key: str, default=None, required=True):
        """Get TES-specific configuration from server config."""
        tes_configs = getattr(
            getattr(self.server_agent_config.server_configs, "comm_configs", None), 
            "tes_configs", 
            {}
        )
        value = tes_configs.get(key, default)
        if required and value is None:
            raise ValueError(f"Required TES configuration '{key}' not found")
        return value
    
    def _create_tes_task(
        self, 
        client_id: str, 
        task_name: str, 
        model_data: Optional[bytes] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a TES task definition for a federated learning client task."""
        task_id = str(uuid.uuid4())
        
        # Prepare inputs
        inputs = []
        if model_data:
            inputs.append({
                "name": "model_data",
                "description": "Serialized model parameters",
                "path": "/tmp/model_data.pkl",
                "type": "FILE",
                "content": model_data.hex()  # Convert bytes to hex string
            })
        
        if metadata:
            inputs.append({
                "name": "task_metadata", 
                "description": "Task metadata",
                "path": "/tmp/metadata.json",
                "type": "FILE",
                "content": json.dumps(metadata)
            })
        
        # Prepare outputs
        outputs = [
            {
                "name": "trained_model",
                "path": "/tmp/output_model.pkl",
                "type": "FILE"
            },
            {
                "name": "training_logs",
                "path": "/tmp/training_logs.json", 
                "type": "FILE"
            }
        ]
        
        # Prepare command
        command = [
            "python", "-m", "appfl.run_tes_client",
            "--task-name", task_name,
            "--client-id", client_id,
            "--model-path", "/tmp/model_data.pkl" if model_data else "",
            "--metadata-path", "/tmp/metadata.json" if metadata else "",
            "--output-path", "/tmp/output_model.pkl",
            "--logs-path", "/tmp/training_logs.json"
        ]
        
        # Create TES task
        tes_task = {
            "name": f"appfl-{task_name}-{client_id}",
            "description": f"APPFL federated learning task: {task_name} for client {client_id}",
            "inputs": inputs,
            "outputs": outputs,
            "executors": [{
                "image": self.docker_image,
                "command": command,
                "workdir": "/tmp"
            }],
            "resources": {
                "cpu_cores": self.resource_requirements["cpu_cores"],
                "ram_gb": self.resource_requirements["ram_gb"], 
                "disk_gb": self.resource_requirements["disk_gb"],
                "preemptible": self.resource_requirements["preemptible"]
            },
            "tags": {
                "appfl_client_id": client_id,
                "appfl_task_name": task_name,
                "appfl_task_id": task_id
            }
        }
        
        return tes_task
    
    def _submit_tes_task(self, tes_task: Dict) -> str:
        """Submit a task to the TES endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        response = requests.post(
            f"{self.tes_endpoint}/ga4gh/tes/v1/tasks",
            json=tes_task,
            headers=headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"TES task submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["id"]
    
    def _get_tes_task_status(self, tes_task_id: str) -> Dict:
        """Get the status of a TES task."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        response = requests.get(
            f"{self.tes_endpoint}/ga4gh/tes/v1/tasks/{tes_task_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"TES task status query failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _wait_for_task_completion(self, tes_task_id: str, client_id: str) -> Tuple[Any, Dict]:
        """Wait for a TES task to complete and return results."""
        while True:
            task_info = self._get_tes_task_status(tes_task_id)
            state = task_info.get("state", "UNKNOWN")
            
            if state == "COMPLETE":
                # Task completed successfully - extract outputs
                outputs = task_info.get("outputs", [])
                model_output = None
                logs_output = {}
                
                for output in outputs:
                    if output.get("name") == "trained_model":
                        # In a real implementation, this would download from storage
                        model_output = output.get("path")  # Placeholder
                    elif output.get("name") == "training_logs":
                        logs_output = json.loads(output.get("content", "{}"))
                
                return model_output, logs_output
            
            elif state in ["SYSTEM_ERROR", "EXECUTOR_ERROR", "CANCELED"]:
                error_msg = f"TES task {tes_task_id} failed with state: {state}"
                if "logs" in task_info:
                    error_msg += f" - Logs: {task_info['logs']}"
                raise RuntimeError(error_msg)
            
            elif state in ["QUEUED", "INITIALIZING", "RUNNING"]:
                self.logger.debug(f"TES task {tes_task_id} for client {client_id} is {state}")
                time.sleep(5)  # Poll every 5 seconds
            
            else:
                self.logger.warning(f"Unknown TES task state: {state}")
                time.sleep(5)
    
    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """Send a task to all clients via TES."""
        if isinstance(metadata, list):
            if len(metadata) != len(self.client_agent_configs):
                raise ValueError("Metadata list length must match number of clients")
            client_metadata = metadata
        else:
            client_metadata = [metadata] * len(self.client_agent_configs)
        
        # Serialize model if needed
        model_bytes = None
        if model is not None:
            if isinstance(model, bytes):
                model_bytes = model
            else:
                import pickle
                model_bytes = pickle.dumps(model)
        
        # Submit tasks to all clients
        futures = []
        for i, client_config in enumerate(self.client_agent_configs):
            client_id = client_config.client_id
            
            # Create and submit TES task
            tes_task = self._create_tes_task(
                client_id, task_name, model_bytes, client_metadata[i]
            )
            
            tes_task_id = self._submit_tes_task(tes_task)
            
            # Submit to thread pool for monitoring
            future = self.executor.submit(
                self._wait_for_task_completion, tes_task_id, client_id
            )
            
            # Register the task
            self._register_task(tes_task_id, future, client_id, task_name)
            futures.append(future)
            
            self.logger.info(f"Submitted TES task {tes_task_id} for client {client_id}")
        
        return futures
    
    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """Send a task to one specific client via TES."""
        # Find client config
        client_config = None
        for config in self.client_agent_configs:
            if config.client_id == client_id:
                client_config = config
                break
        
        if client_config is None:
            raise ValueError(f"Client {client_id} not found in configurations")
        
        # Serialize model if needed
        model_bytes = None
        if model is not None:
            if isinstance(model, bytes):
                model_bytes = model
            else:
                import pickle
                model_bytes = pickle.dumps(model)
        
        # Create and submit TES task
        tes_task = self._create_tes_task(client_id, task_name, model_bytes, metadata)
        tes_task_id = self._submit_tes_task(tes_task)
        
        # Submit to thread pool for monitoring
        future = self.executor.submit(
            self._wait_for_task_completion, tes_task_id, client_id
        )
        
        # Register the task
        self._register_task(tes_task_id, future, client_id, task_name)
        
        self.logger.info(f"Submitted TES task {tes_task_id} for client {client_id}")
        return future
    
    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """Receive results from all clients with running tasks."""
        client_results = {}
        client_metadata = {}
        
        # Wait for all executing tasks to complete
        completed_futures = []
        for future, task_id in list(self.executing_task_futs.items()):
            if future.done():
                completed_futures.append((future, task_id))
        
        # If no completed futures, wait for at least one
        if not completed_futures and self.executing_task_futs:
            completed_futures = [(next(as_completed(self.executing_task_futs.keys())), None)]
        
        for future, task_id in completed_futures:
            if task_id is None:
                task_id = self.executing_task_futs[future]
            
            task_info = self.executing_tasks[task_id]
            client_id = task_info.client_id
            
            try:
                model, metadata = future.result()
                client_results[client_id] = model
                client_metadata[client_id] = metadata
                
                # Clean up
                del self.executing_tasks[task_id]
                del self.executing_task_futs[future]
                
            except Exception as e:
                self.logger.error(f"Task {task_id} for client {client_id} failed: {e}")
                client_results[client_id] = None
                client_metadata[client_id] = {"error": str(e)}
        
        return client_results, client_metadata
    
    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """Receive result from the first client that completes."""
        if not self.executing_task_futs:
            raise RuntimeError("No executing tasks")
        
        # Wait for first completion
        completed_future = next(as_completed(self.executing_task_futs.keys()))
        task_id = self.executing_task_futs[completed_future]
        task_info = self.executing_tasks[task_id]
        client_id = task_info.client_id
        
        try:
            model, metadata = completed_future.result()
            
            # Clean up
            del self.executing_tasks[task_id] 
            del self.executing_task_futs[completed_future]
            
            return client_id, model, metadata
            
        except Exception as e:
            self.logger.error(f"Task {task_id} for client {client_id} failed: {e}")
            return client_id, None, {"error": str(e)}
    
    def cancel_all_tasks(self):
        """Cancel all running TES tasks."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        for task_id in list(self.executing_tasks.keys()):
            try:
                response = requests.post(
                    f"{self.tes_endpoint}/ga4gh/tes/v1/tasks/{task_id}:cancel",
                    headers=headers
                )
                if response.status_code == 200:
                    self.logger.info(f"Cancelled TES task {task_id}")
                else:
                    self.logger.warning(f"Failed to cancel TES task {task_id}: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error cancelling TES task {task_id}: {e}")
        
        # Cancel futures
        for future in self.executing_task_futs.keys():
            future.cancel()
        
        # Clear tracking
        self.executing_tasks.clear()
        self.executing_task_futs.clear()
    
    def shutdown_all_clients(self):
        """Shutdown the TES communicator."""
        self.cancel_all_tasks()
        self.executor.shutdown(wait=True)
        self.logger.info("TES Server Communicator shutdown complete")