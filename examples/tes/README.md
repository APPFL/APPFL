# APPFL GA4GH TES Integration

This directory contains the TES (Task Execution Service) integration for APPFL, following the same architectural patterns as Ray and Globus Compute communicators.

## Overview

The TES integration allows APPFL to execute federated learning clients as containerized tasks on GA4GH TES-compliant infrastructures, providing:

- **Cross-platform execution**: Run clients on different compute environments
- **Standardized task management**: Use GA4GH TES API for task submission and monitoring  
- **Resource specification**: Define CPU, RAM, and disk requirements per client
- **Fault tolerance**: Built-in task retry and error handling
- **APPFL architectural consistency**: Follows established patterns from other communicators

## Prerequisites

### 1. TES Server
You need access to a GA4GH TES-compliant server:
- **Funnel** (recommended for testing): Local TES server
- **TESK**: TES on Kubernetes  
- **Microsoft TES**: TES on Azure
- Cloud provider TES implementations

### 2. Docker Image
Build the APPFL client Docker image:
```bash
cd examples/tes
docker build -t appfl/client:latest .
```

### 3. Authentication (Optional)
Set authentication for your TES endpoint:
```bash
export TES_AUTH_TOKEN="your_access_token"
```

## Quick Start

### 1. Install and Start Funnel (Local Testing)
```bash
# Download Funnel
curl -L https://github.com/ohsu-comp-bio/funnel/releases/download/0.10.1/funnel-linux-amd64-0.10.1.tar.gz | tar xz
sudo mv funnel /usr/local/bin/

# Start TES server
funnel server run &
```

### 2. Build Docker Image
```bash
cd examples/tes
docker build -t appfl/client:latest .
```

### 3. Run Integration Test
```bash
python test_tes_integration.py
```

### 4. Run Federated Learning Test
```bash
python test_tes_federated.py
```

## Configuration

### Server Configuration
```yaml
server_configs:
  scheduler: "SyncScheduler"
  aggregator: "FedAvgAggregator" 
  num_global_epochs: 10
  comm_configs:
    tes_configs:
      tes_endpoint: "http://localhost:8000"
      auth_token: "${TES_AUTH_TOKEN}"
      docker_image: "appfl/client:latest"
      resource_requirements:
        cpu_cores: 2
        ram_gb: 4.0
        disk_gb: 20.0
        preemptible: false
```

### Client Configuration
```yaml
client_configs:
  train_configs:
    trainer: "VanillaTrainer"
    num_local_steps: 100
    optim: "Adam"
  model_configs:
    model: "CNN"
  data_configs:
    dataset: "MNIST"
    batch_size: 64
```

## Usage

### Python Code
```python
from appfl.agent import ServerAgent
from appfl.comm.tes import TESServerCommunicator
from appfl.config import ServerAgentConfig, ClientAgentConfig

# Load configurations
server_config = ServerAgentConfig(**config_dict)
client_configs = [ClientAgentConfig(**client_config) for client_config in client_configs_list]

# Create server agent and TES communicator
server_agent = ServerAgent(server_config)
tes_comm = TESServerCommunicator(server_config, client_configs)

# Run federated learning
for epoch in range(num_epochs):
    global_model = server_agent.get_parameters()
    
    # Submit training tasks to TES
    tes_comm.send_task_to_all_clients("train", model=global_model)
    
    # Collect results and aggregate
    results, metadata = tes_comm.recv_result_from_all_clients()
    for client_id, local_model in results.items():
        server_agent.global_update(client_id, local_model)
```

## Files

- **`test_tes_integration.py`**: Basic integration test (no TES server required)
- **`test_tes_federated.py`**: Complete federated learning test (requires TES server)
- **`Dockerfile`**: APPFL client container definition
- **`test_setup.sh`**: Automated setup script for Funnel

## Testing

### Phase 1: Basic Integration Test (No Dependencies)
```bash
python test_tes_integration.py
```
Tests module imports, communicator creation, and task structure. **No TES server or Docker image required.**

### Phase 2: Federated Learning Test (Full Setup Required)
```bash
# Start Funnel server
funnel server run &

# Build Docker image  
docker build -t appfl/client:latest .

# Run federated learning test
python test_tes_federated.py
```
Tests complete federated learning workflow with actual TES task submission.

## Troubleshooting

### Common Issues

1. **TES server unreachable**
   - Check `funnel server run` is running
   - Verify endpoint URL: `curl http://localhost:8000/ga4gh/tes/v1/service-info`

2. **Docker image not found**
   - Build image: `docker build -t appfl/client:latest .`
   - Check image exists: `docker images | grep appfl/client`

3. **Task execution failures**
   - Check TES server logs
   - Verify Docker image contains APPFL installation
   - Check resource requirements vs available resources

4. **Import errors**
   - Ensure APPFL is installed: `pip install -e ".[examples]"`
   - Verify Python path includes APPFL source

## Architecture

The TES integration follows APPFL's established communicator patterns:

- **TESServerCommunicator**: Inherits from `BaseServerCommunicator`, manages TES task submission
- **TESClientCommunicator**: Handles container-side task execution
- **Integration**: Compatible with existing APPFL agents, configurations, and workflows

This ensures consistency with other APPFL communicators (Ray, Globus Compute, gRPC, MPI) while providing TES-specific functionality.