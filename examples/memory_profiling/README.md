# APPFL gRPC Memory Profiling and Optimization [Experimental]

This directory contains tools for memory profiling and optimization of APPFL's federated learning components using memray.

## Quick Start

1. **Install memray**:
   ```bash
   pip install memray
   ```

2. **Run memory profiling experiments (Recommended)**:
   
   **CIFAR-10 experiment** (**recommended** - uses real CIFAR-10 configs):
   ```bash
   cd examples
   chmod +x memory_profiling/run_cifar_experiment.sh
   ./memory_profiling/run_cifar_experiment.sh
   ```
   
   **ResNet experiment** (alternative - focuses on training memory with dummy data):
   ```bash
   cd examples
   chmod +x memory_profiling/run_resnet_experiment.sh
   ./memory_profiling/run_resnet_experiment.sh
   ```
   
   **MNIST experiment** (alternative - uses real MNIST data):
   ```bash
   cd examples
   chmod +x memory_profiling/run_mnist_experiment.sh
   ./memory_profiling/run_mnist_experiment.sh
   ```

3. **Or run experiments manually**:
   ```bash
   cd examples
   
   # Original version
   python memory_profiling/run_server_memray.py --config ./memory_profiling/configs/server_resnet_dummy.yaml &
   python memory_profiling/run_client_memray.py --config ./memory_profiling/configs/client_1_resnet_dummy.yaml &
   python memory_profiling/run_client_memray.py --config ./memory_profiling/configs/client_2_resnet_dummy.yaml
    
   
   # Optimized version
   python memory_profiling/run_server_memray.py --config ./memory_profiling/configs/server_resnet_dummy.yaml --use_optimized_version &
   python memory_profiling/run_client_memray.py --config ./memory_profiling/configs/client_1_resnet_dummy.yaml --use_optimized_version &
   python memory_profiling/run_client_memray.py --config ./memory_profiling/configs/client_2_resnet_dummy.yaml --use_optimized_version
   ```

4. **View results**: Open the generated HTML flamegraph files in your browser from the output directory.

## Files Overview

- `run_server_memray.py` - Memory profiling wrapper for gRPC server
- `run_client_memray.py` - Memory profiling wrapper for gRPC client  
- `run_cifar_experiment.sh` - CIFAR-10 memory profiling experiment (recommended)
- `run_resnet_experiment.sh` - ResNet memory profiling experiment
- `run_mnist_experiment.sh` - MNIST memory profiling experiment
- `analyze_profiles.py` - Automatic analysis script for memory profiles
- `dummy_cifar10_dataset.py` - Lightweight dummy dataset for training memory isolation
- `configs/` - Configuration files for ResNet experiments

## Memory Optimizations Implemented

APPFL now includes built-in memory optimizations that can be enabled with the `--use_optimized_version` flag. These optimizations are integrated directly into the main codebase with `optimize_memory` configuration flags.

### Components Optimized

#### 1. VanillaTrainer
- **Tensor Cloning**: Uses `tensor.clone().detach()` instead of `copy.deepcopy()` for model state storage
- **In-place Operations**: Memory-efficient gradient computation
- **Strategic Cleanup**: Immediate deletion of previous model states after use

#### 2. ServerAgent & ClientAgent
- **Model Deserialization**: Context managers for efficient BytesIO handling
- **CPU-first Loading**: Reduces GPU memory pressure
- **Garbage Collection**: Strategic cleanup after model processing and training

#### 3. gRPC Communicators (Server & Client)
- **Efficient Byte Handling**: Uses `bytearray` instead of bytes concatenation
- **Streaming Optimization**: Periodic garbage collection during large transfers
- **Context Managers**: Memory-efficient model serialization/deserialization

### Configuration

Memory optimizations are controlled by `optimize_memory` flags in configuration:

```yaml
# Server configuration
server_configs:
  optimize_memory: true
  comm_configs:
    grpc_configs:
      optimize_memory: true

# Client configuration  
client_configs:
  optimize_memory: true
  train_configs:
    optimize_memory: true
```

Or automatically enabled with `--use_optimized_version` flag in profiling scripts.

### Expected Benefits

- **Reduced Peak Memory Usage**: 20-40% reduction in memory footprint
- **Faster Garbage Collection**: Strategic cleanup reduces GC pressure
- **Better Large Model Handling**: Efficient tensor operations for models like ResNet
- **Improved gRPC Performance**: Optimized streaming for large parameter transfers

### Viewing Memory Profiles

Use memray to analyze the generated profiles:

```bash
# Generate flamegraph
memray flamegraph memory_profiles/server_optimized_memory_profile.bin

# View memory statistics  
memray stats memory_profiles/client_Client1_optimized_memory_profile.bin

# Compare original vs optimized
memray stats memory_profiles/server_original_memory_profile.bin
memray stats memory_profiles/server_optimized_memory_profile.bin
```