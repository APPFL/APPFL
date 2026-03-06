# DIMAT Integration into APPFL

This document describes the changes made to integrate the DIMAT (Decentralized Iterative Merging-And-Training) algorithm into the APPFL framework, and the validation experiments confirming correctness.

## Overview

DIMAT is a decentralized federated learning algorithm that aligns models in feature space using activation matching before merging, rather than naively averaging weights. It supports multiple topologies (fully connected, ring, bipartite) and uses the Hungarian algorithm for permutation-based alignment.

The integration adds DIMAT as a pluggable aggregator and trainer within APPFL's existing architecture, requiring no changes to the core framework.

## Files Added

### Core Algorithm Components

| File | Description |
|------|-------------|
| `src/appfl/algorithm/aggregator/dimat_aggregator.py` | Server-side aggregator implementing DIMAT merge logic for FC and ring topologies. Manages proxy data loading, per-agent neighbor merging, BN reset, and topology weight computation. |
| `src/appfl/algorithm/trainer/dimat_trainer.py` | Client-side trainer extending `VanillaTrainer`. Preserves server-computed BN statistics when loading merged parameters. |

### DIMAT Utility Modules (`src/appfl/algorithm/aggregator/dimat_utils/`)

| File | Description |
|------|-------------|
| `base_graph.py` | Abstract `BIGGraph` class representing computation graphs for neural networks. Handles hook registration, intermediate activation capture, and feature reshaping. |
| `resnet_graph.py` | Graph construction for ResNet architectures (ResNet20, ResNet18, ResNet50). Maps layers, skip connections, and shortcuts into a DAG for alignment. |
| `vgg_graph.py` | Graph construction for VGG architectures (VGG11, VGG16). |
| `matching_functions.py` | Feature-space alignment algorithms: `match_tensors_permute` (Hungarian algorithm), `match_tensors_zipit`, `match_tensors_optimal`, and `match_tensors_identity`. Each computes merge/unmerge transformation matrices. |
| `metric_calculators.py` | Pairwise alignment metrics (`CovarianceMetric`, `MeanMetric`) computed over intermediate activations to guide the matching algorithms. |
| `model_merger.py` | `ModelMerge` class orchestrating the full pipeline: compute metrics, compute transformations, apply transformations, and produce the merged state dict with interpolation weights. |
| `am_utils.py` | Utilities: `SpaceInterceptor` module for controlling transformations over residual connections, `reset_bn_stats()` for recomputing BatchNorm statistics using mixed-precision (`autocast`). |
| `__init__.py` | Package init (empty). |

### Configuration Files (`examples/resources/configs/cifar100/`)

| File | Description |
|------|-------------|
| `server_dimat_ring.yaml` | Server config: `DIMATaggregator` with ring topology, `resnet20` graph, `match_tensors_permute` matching, proxy dataset path, 100 global rounds. |
| `client_dimat_ring.yaml` | Client config: `DIMATTrainer`, Adam optimizer (lr=0.001), 2 local epochs, batch size 100, CIFAR-100 dataset with class-balanced IID partition. |

### Dataset Loaders (`examples/resources/dataset/`)

| File | Description |
|------|-------------|
| `cifar100_dimat.py` | CIFAR-100 client dataset with **class-balanced IID partitioning** and **lazy transforms** (augmentation applied fresh on every access). |
| `cifar100_proxy_norm.py` | Full CIFAR-100 training set (50k images) as proxy data for BN reset and activation statistics, with **lazy transforms**. |

### Model Definition (`examples/resources/model/`)

| File | Description |
|------|-------------|
| `resnet20_dimat.py` | ResNet20 with width multiplier w=8 (channels 128/256/512), 3x3 shortcut convolutions, and `kaiming_normal` initialization, matching the DIMAT paper. |

### Execution Scripts

| File | Description |
|------|-------------|
| `examples/mpi/run_mpi_dimat_pretrain.py` | MPI orchestration with two phases: (1) local pre-training for N epochs, (2) standard FL merge-train loop. |
| `test_mpi_dimat_cifar100_ring_pretrain.sh` | SLURM batch script for the full CIFAR-100 Ring experiment (5 agents, 100 pre-train epochs, 100 merge-train rounds). |

### Modified Existing Files

| File | Change |
|------|--------|
| `src/appfl/algorithm/aggregator/__init__.py` | Added `DIMATaggregator` import (guarded by try-except). |
| `src/appfl/algorithm/trainer/__init__.py` | Added `DIMATTrainer` import (guarded by try-except). |

## Key Design Decisions

### Lazy Data Augmentation

The original DIMAT code uses `torchvision.datasets.CIFAR100` directly, which applies random transforms (horizontal flip, random crop) fresh on every `__getitem__` call. APPFL's built-in `Dataset` class stores pre-computed tensors, which would freeze augmentation to a single random realization.

Our dataset loaders return `torch.utils.data.Subset` wrappers around the raw torchvision dataset, preserving lazy transform behavior. This was critical for matching paper accuracy.

### Class-Balanced IID Partitioning

APPFL's built-in `iid_partition()` uses `np.array_split(range(N), num_clients)`, which creates contiguous index blocks. Since CIFAR-100 is ordered by class, this produces a non-IID split where each client sees only 20 of 100 classes.

Our partition matches the original DIMAT `DataPartition`: for each of the 100 classes, distribute samples evenly across all 5 clients, ensuring every client has equal representation of every class.

### Server-Side BN Reset Only

After DIMAT merges models via activation matching, BatchNorm running statistics must be recomputed. The server resets BN using the **full 50k proxy dataset**, which captures the complete data distribution. The client trainer (`DIMATTrainer`) deliberately does **not** re-reset BN with its local 10k partition, as this would overwrite the higher-quality server statistics and degrade accuracy.

### Mixed-Precision (autocast)

The original DIMAT code uses `torch.cuda.amp.autocast()` during BN reset and intermediate activation computation. We match this in `am_utils.py:reset_bn_stats()` and `base_graph.py:compute_intermediates()` using the modern `torch.amp.autocast()` API.

## Validation Experiment

### Setup

- **Dataset**: CIFAR-100
- **Model**: ResNet20 (w=8, channels 128/256/512, 100 output classes)
- **Agents**: 5
- **Topology**: Ring (each agent merges with 2 adjacent neighbors, weights 1/3 each)
- **Partition**: IID (class-balanced)
- **Pre-training**: 100 epochs per agent (local, independent)
- **Merge-train**: 100 rounds, 2 local epochs per round
- **Optimizer**: Adam, lr=0.001 (fresh each round)
- **Batch size**: 100
- **Paper target**: 67.12 +/- 0.22% (Table 2, DIMAT Ring)

### Results

The APPFL integration was validated against both the paper target and a direct run of the original DIMAT codebase:

| Implementation | Peak Pre-Validation Accuracy |
|----------------|------------------------------|
| Paper (reported) | 67.12 +/- 0.22% |
| Original DIMAT code (our run) | ~67.9% |
| **APPFL integration** | **~67.9%** |

Accuracy progression of the APPFL integration (average across 5 agents):

| Round | Avg Accuracy |
|-------|-------------|
| 1 | 48.8% |
| 3 | 50.1% |
| 5 | 58.9% |
| 7 | 64.0% |
| 9 | 65.9% |
| 10 | 66.9% |
| 11 | 67.8% |
| 12 | **67.9%** |

The APPFL integration matches the original DIMAT implementation and meets the paper's reported accuracy.

### Reproducing the Experiment

```bash
# Ensure APPFL is installed with MPI support
pip install -e ".[dev,examples,mpi]"

# Submit the SLURM job (requires GPU node with MPI)
sbatch test_mpi_dimat_cifar100_ring_pretrain.sh

# Or run directly with MPI
cd examples
mpirun --oversubscribe -n 6 python mpi/run_mpi_dimat_pretrain.py \
    --server_config ../examples/resources/configs/cifar100/server_dimat_ring.yaml \
    --client_config ../examples/resources/configs/cifar100/client_dimat_ring.yaml \
    --pretrain_epochs 100
```

## Bugs Found and Fixed During Integration

During development, several subtle discrepancies between the APPFL integration and the original DIMAT code caused significant accuracy degradation. Each was identified through systematic line-by-line comparison and ablation:

1. **Frozen data augmentation** (59% -> 67%): APPFL's `Dataset` class pre-computes transforms once at construction. Fixed by returning `Subset` wrappers around raw torchvision datasets to preserve lazy per-access augmentation.

2. **Non-IID partitioning** (contributing to 59% plateau): Sequential index splitting (`np.array_split`) on class-ordered CIFAR-100 gave each client only 20/100 classes. Fixed with class-balanced distribution matching the original.

3. **Client-side BN re-reset** (57.7% -> 59%): `DIMATTrainer.load_parameters()` was overwriting server BN statistics (computed from 50k proxy data) with local 10k statistics. Fixed by removing the re-reset.

4. **Missing autocast** (minor): Added `torch.amp.autocast()` to `reset_bn_stats()` and `compute_intermediates()` to match original mixed-precision behavior.
