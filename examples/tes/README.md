# APPFL GA4GH TES Integration Example

This directory contains examples for running APPFL (Advanced Privacy-Preserving Federated Learning) using GA4GH Task Execution Service (TES) for distributed client execution.

## Overview

The GA4GH TES integration allows APPFL to execute federated learning clients as containerized tasks on TES-compliant compute infrastructures. This enables:

- **Cross-platform execution**: Run clients on different compute environments (HPC, cloud, edge)
- **Standardized task management**: Use GA4GH TES API for task submission, monitoring, and management  
- **Resource specification**: Define CPU, RAM, and disk requirements per client task
- **Fault tolerance**: Built-in task retry and error handling through TES
- **Scalability**: Support for large-scale federated learning deployments

## Prerequisites

### 1. TES Server
You need access to a GA4GH TES-compliant server. Options include:
- [Funnel](https://ohsu-comp-bio.github.io/funnel/) - Local TES server
- [TESK](https://github.com/EMBL-EBI-TSI/TESK) - TES on Kubernetes
- [Microsoft TES](https://github.com/microsoft/ga4gh-tes) - TES on Azure
- Cloud provider TES implementations

### 2. Docker Image
Build an APPFL client Docker image or use a pre-built one:

```dockerfile
FROM python:3.8-slim

# Install APPFL and dependencies
RUN pip install appfl[examples]

# Copy client configuration
COPY client_tes.yaml /app/client_config.yaml

# Set working directory
WORKDIR /app

# Entry point
CMD ["python", "-m", "appfl.run_tes_client"]
```

### 3. Authentication
Set up authentication for your TES endpoint:
```bash
export TES_AUTH_TOKEN="your_access_token"
```

## Configuration

### Server Configuration (`server_tes.yaml`)

```yaml
server_configs:
  comm_configs:
    tes_configs:
      tes_endpoint: "https://your-tes-endpoint.org"
      auth_token: "${TES_AUTH_TOKEN}"
      docker_image: "appfl/client:latest"
      resource_requirements:
        cpu_cores: 2
        ram_gb: 4.0
        disk_gb: 20.0
        preemptible: false
```

### Client Configuration (`client_tes.yaml`)

Configuration for clients running in TES containers:

```yaml
client_id: "tes_client_1"
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

### 1. Start TES Server
Run the federated learning server that submits tasks to TES:

```bash
cd examples/tes
python run_server.py --config ../resources/configs/mnist/server_tes.yaml --num-clients 4
```

### 2. Monitor Execution
The server will:
1. Submit training tasks to the TES endpoint
2. Monitor task execution status
3. Collect results when tasks complete
4. Perform model aggregation
5. Repeat for subsequent rounds

### 3. View Logs
Check server logs for execution details:
- Task submission confirmations
- TES task IDs and status updates
- Client results and aggregation progress
- Error messages and debugging information

## TES Task Structure

Each federated learning round creates TES tasks with:

**Inputs:**
- Serialized global model parameters
- Task metadata (epoch, client_id, etc.)
- Client configuration

**Executors:**
- Docker container with APPFL environment
- Training command with appropriate arguments
- Resource requirements (CPU, RAM, disk)

**Outputs:**
- Updated local model parameters
- Training logs and metrics
- Client metadata

## Error Handling

The TES integration includes robust error handling:

- **Task failures**: Automatic retry logic for failed tasks
- **Network issues**: Connection retries for TES API calls
- **Resource constraints**: Graceful handling of insufficient resources
- **Timeout management**: Configurable timeouts for long-running tasks

## Monitoring and Debugging

### TES Task Status
Tasks can be in various states:
- `QUEUED`: Waiting for resources
- `INITIALIZING`: Setting up execution environment
- `RUNNING`: Actively executing
- `COMPLETE`: Successfully finished
- `EXECUTOR_ERROR`: Task execution failed
- `SYSTEM_ERROR`: Infrastructure issue
- `CANCELED`: Manually cancelled

### Debugging Tips

1. **Check TES endpoint connectivity**:
   ```bash
   curl -X GET https://your-tes-endpoint.org/ga4gh/tes/v1/service-info
   ```

2. **Verify authentication**:
   ```bash
   curl -H "Authorization: Bearer $TES_AUTH_TOKEN" https://your-tes-endpoint.org/ga4gh/tes/v1/tasks
   ```

3. **Monitor task logs**:
   ```python
   # In your server script
   task_info = tes_communicator._get_tes_task_status(task_id)
   print(task_info.get('logs', []))
   ```

4. **Resource requirements**:
   Ensure your TES server has sufficient resources for the requested CPU, RAM, and disk requirements.

## Advanced Features

### Custom Docker Images
Build specialized images for different federated learning scenarios:

```dockerfile
# GPU-enabled image
FROM nvidia/cuda:11.0-runtime-ubuntu20.04
RUN pip install appfl[examples] torch torchvision

# HPC-optimized image  
FROM intel/oneapi-hpckit
RUN pip install appfl[examples,mpi]

# Privacy-focused image
FROM python:3.8-slim
RUN pip install appfl[examples] opacus
```

### Resource Optimization
Tune resource requirements based on:
- Model size and complexity
- Dataset size per client
- Training algorithm requirements
- Available infrastructure capacity

### Fault Tolerance
Implement additional fault tolerance:
- Client checkpoint/resume capability
- Dynamic client replacement
- Partial aggregation strategies
- Network partition handling

## Troubleshooting

**Common Issues:**

1. **TES endpoint unreachable**
   - Check network connectivity
   - Verify endpoint URL format
   - Confirm firewall/proxy settings

2. **Authentication failures**
   - Validate auth token format
   - Check token expiration
   - Verify permissions scope

3. **Resource unavailable**
   - Reduce resource requirements
   - Check cluster capacity
   - Consider preemptible instances

4. **Docker image issues**
   - Verify image exists and is accessible
   - Check image tags and versions
   - Ensure proper registry credentials

5. **Task execution failures**
   - Review task logs for error details
   - Check input data validity
   - Verify client configuration

## References

- [GA4GH TES Specification](https://ga4gh.github.io/task-execution-schemas/docs/)
- [APPFL Documentation](https://appfl.readthedocs.io/)
- [GA4GH Cloud Work Stream](https://www.ga4gh.org/work_stream/cloud/)