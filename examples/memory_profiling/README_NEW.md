# Memory Optimization using Chunked Model Transmission and Aggregation

To further optimize memory usage during federated learning with large models, we have implemented chunked model transmission and aggregation. This approach breaks down large model parameters into smaller chunks, reducing peak memory consumption on both the server and client sides.

## How to Run

You can easily leverage the script provided in `examples/memory_profiling/run_llm_experiment.sh` to execute the memory profiling experiments for large language models (LLMs) with and without chunked model transmission. 

```bash
cd examples
chmod +x memory_profiling/run_llm_experiment.sh
./memory_profiling/run_llm_experiment.sh
```

## Explanation of the Script

### Configurations

The script launches one server and two clients and runs the optimized version of FL training on LLM. The server uses the configuration file `examples/memory_profiling/configs/server_llm_dummy.yaml`, as shown below, which uses a newly defined dummy LLM trainer ([`DummyLLMTrainer`](../../src/appfl/algorithm/trainer/llm_dummy_trainer.py) - it does nothing but loading and returning the large LLM). To enable the new optimization, you need to set `use_model_chunking: True` in the gRPC communication configurations for both the server and clients. The model used in this example is Meta's Llama 3.1 8B model, which is loaded via huggingface using the script in [`examples/resources/model/hf_llm.py`](../resources/model/hf_llm.py).

```yaml
# configs/server_llm_dummy.yaml
client_configs:
  train_configs:
    trainer: "LLMDummyTrainer"

  model_configs:
    model_path: "./resources/model/hf_llm.py"
    model_name: "load_hf_llm"
    model_kwargs:
      model_name: "meta-llama/Llama-3.1-8B"

server_configs:
  num_clients: 2
  scheduler: "SyncScheduler"
  scheduler_kwargs:
    same_init_model: True
  aggregator: "FedAvgAggregator"
  aggregator_kwargs:
    client_weights_mode: "equal"
  device: "cpu"
  num_global_epochs: 3
  logging_output_dirname: "./output"
  logging_output_filename: "result"
  comm_configs:
    grpc_configs:
      server_uri: localhost:50051
      max_message_size: 104857600  # 100MB for LLM parameters
      use_ssl: False
      use_model_chunking: True # Enable chunked model transmission
```

As for the configuration, its data configurations are not very relevant for this case as the dummy trainer does not use any data. The clients use the configuration files `configs/client_1_llm_dummy.yaml` and `configs/client_2_llm_dummy.yaml`, which are similar to the server configuration in terms of enabling chunked model transmission. It should be noted that the `use_model_chunking` flag must be consistently set to `True` to enable chunked model transmission and aggregation.

```yaml
# configs/client_1_resnet_dummy.yaml
client_id: "Client1"
train_configs:
  # Device
  device: "cpu"  # Use CPU for consistent memory profiling
  # Logging and outputs
  logging_output_dirname: "./output"
  logging_output_filename: "result"

# Dummy dataset configuration
data_configs:
  dataset_path: "./memory_profiling/dummy_cifar10_dataset.py"
  dataset_name: "get_dummy_cifar10"
  dataset_kwargs:
    num_clients: 2
    client_id: 0
    samples_per_client: 64  # Very small dataset to isolate training memory

comm_configs:
  grpc_configs:
    server_uri: localhost:50051
    max_message_size: 104857600  # 10MB for ResNet parameters
    use_ssl: False
    use_model_chunking: True # Enable chunked model transmission
```

### Output

The script will generate memory profiling files for the optimized version with chunked model transmission. 