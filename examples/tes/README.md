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

### 3. Run Federated Learning
```bash
# Using default configuration files
python run.py

# Using custom configuration files
python run.py --server_config ./server.yaml --client_config ./clients.yaml

# With authentication token
python run.py --server_config ./server.yaml --client_config ./clients.yaml --auth_token "your_tes_token"
```

## Configuration

### Basic Configuration (Single TES Endpoint)
```yaml
server_configs:
  comm_configs:
    tes_configs:
      tes_endpoint: "http://localhost:8000"
      auth_token: "${TES_AUTH_TOKEN}"
      docker_image: "appfl/client:latest"
      resource_requirements:
        cpu_cores: 2
        ram_gb: 4.0
        disk_gb: 20.0
```

### Multi-Institute Configuration (Multiple TES Endpoints)
```yaml
server_configs:
  comm_configs:
    tes_configs:
      # Default settings
      tes_endpoint: "http://localhost:8000"
      docker_image: "appfl/client:latest"

client_configs:
  - client_id: "institute_a_client"
    # Override with Institute A's TES endpoint
    tes_endpoint: "https://tes.institute-a.edu"
    auth_token: "${INSTITUTE_A_TOKEN}"
    docker_image: "institute-a.edu/appfl-client:v1.0"
    
  - client_id: "institute_b_client"  
    # Override with Institute B's TES endpoint
    tes_endpoint: "https://tes.institute-b.edu"
    auth_token: "${INSTITUTE_B_TOKEN}"
    docker_image: "institute-b.edu/appfl-client:v1.0"
    
  - client_id: "local_client"
    # Uses default TES endpoint (localhost:8000)
```

See `multi_tes_example.yaml` for a complete multi-institute configuration example.

### Data Access Configuration

**Volume Mount (Local Data)**:
```yaml
client_configs:
  - client_id: "hospital_client"
    data_configs:
      volume_mounts:
        - name: "hospital_data"
          host_path: "/secure/hospital/data"  # Host directory
          container_path: "/data"             # Container mount
          read_only: true                     # Security
      environment:
        DATA_PATH: "/data"
        HOSPITAL_ID: "hospital_a"
```

**Pre-built Data Container**:
```yaml
client_configs:
  - client_id: "institute_client"
    docker_image: "institute.edu/appfl-data:v1.0"  # Contains data
    data_configs:
      environment:
        DATA_SOURCE: "embedded"
        DATA_PATH: "/app/data"
```

**Cloud Storage Access**:
```yaml
client_configs:
  - client_id: "cloud_client"
    data_configs:
      environment:
        DATA_SOURCE: "s3"
        S3_BUCKET: "private-fl-data"
        AWS_ACCESS_KEY_ID: "${AWS_KEY}"
```

See `data_volume_example.yaml` for complete data access patterns.

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

- **`run.py`**: Main federated learning script (like examples/globus_compute/run.py)
- **`server.yaml`**: Server configuration for TES federated learning
- **`clients.yaml`**: Client configurations for TES federated learning
- **`Dockerfile`**: APPFL client container definition

## Usage

Run federated learning with TES:

```bash
# Start TES server (Funnel)
funnel server run &

# Build Docker image  
docker build -t appfl/client:latest .

# Run federated learning
python run.py --server_config ./server.yaml --client_config ./clients.yaml
```

For different scenarios, just use different YAML files:
- **Multi-TES endpoints**: Different `clients.yaml` with per-client TES endpoints
- **Data mounting**: Modified `clients.yaml` with volume mount configurations  
- **Different algorithms**: Modified `server.yaml` with different aggregators/schedulers

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