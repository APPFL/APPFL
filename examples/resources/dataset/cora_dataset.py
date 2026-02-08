import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List


class NodeSubgraphDataset(Dataset):
    """
    Custom dataset that wraps PyG Data objects for node-level tasks in federated learning.
    
    Each client gets a subset of nodes from the graph, along with their local neighborhoods.
    This enables federated node classification where each client trains on their local
    portion of the graph structure.
    """
    
    def __init__(self, data: Data, node_indices: torch.Tensor):
        """
        Args:
            data: PyG Data object containing the full graph
            node_indices: Indices of nodes assigned to this client
        """
        self.data = data
        self.node_indices = node_indices
        
    def __len__(self):
        return len(self.node_indices)
    
    def __getitem__(self, idx):
        """
        Returns the features and label for a specific node.
        Note: In practice, GNN training uses the full graph structure,
        but we track which nodes belong to each client for training.
        """
        node_idx = self.node_indices[idx]
        return self.data.x[node_idx], self.data.y[node_idx], node_idx


def get_cora(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "random",
    alpha: float = 0.5,
    **kwargs
) -> Tuple[NodeSubgraphDataset, NodeSubgraphDataset]:
    """
    Load and partition the Cora dataset for federated learning.
    
    Args:
        num_clients: Total number of clients in the federated learning setup
        client_id: ID of the current client (0-indexed)
        partition_strategy: Strategy for partitioning nodes among clients
            - "random": Random partitioning of nodes
            - "label_skew": Non-IID partitioning based on class distribution
        alpha: Concentration parameter for Dirichlet distribution (for label_skew)
               Lower values (e.g., 0.1) create more non-IID splits
               Higher values (e.g., 1.0) create more balanced splits
        **kwargs: Additional arguments
    
    Returns:
        train_dataset: Training dataset for the client
        val_dataset: Validation dataset for the client (shared test set)
    """
    # Download directory for the dataset
    data_dir = os.path.join(os.getcwd(), "datasets", "Cora")
    
    # Load the Cora dataset using PyTorch Geometric
    # The dataset is split into train/val/test by the standard split masks
    dataset = Planetoid(root=data_dir, name="Cora")
    data = dataset[0]  # Cora has only one graph
    
    # Get train, val, and test masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    # Get indices for training nodes
    train_indices = torch.where(train_mask)[0]
    val_indices = torch.where(val_mask)[0]
    test_indices = torch.where(test_mask)[0]
    
    # Partition training nodes among clients
    if partition_strategy == "random":
        # Random partitioning: shuffle and split evenly
        perm = torch.randperm(len(train_indices))
        train_indices = train_indices[perm]
        
        # Split train indices into num_clients chunks
        client_train_indices = torch.chunk(train_indices, num_clients)[client_id]
        
    elif partition_strategy == "label_skew":
        # Non-IID partitioning using Dirichlet distribution
        # This creates label imbalance across clients, simulating real-world scenarios
        
        num_classes = dataset.num_classes
        train_labels = data.y[train_indices].numpy()
        
        # Get minimum size per client
        min_size = 0
        N = len(train_indices)
        
        # Use Dirichlet distribution to create non-IID splits
        client_indices_list = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            # Get indices for this class
            idx_k = train_indices[train_labels == k].numpy()
            np.random.shuffle(idx_k)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance proportions to ensure each client gets data
            proportions = np.array([p * (len(idx_k) / num_clients) for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Split the indices based on proportions
            splits = np.split(idx_k, proportions)
            for i, split in enumerate(splits):
                if i < num_clients:
                    client_indices_list[i].extend(split)
        
        # Convert to tensor
        client_train_indices = torch.tensor(client_indices_list[client_id], dtype=torch.long)
        
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")
    

    client_val_indices = test_indices
    
    # Create datasets for this client
    train_dataset = NodeSubgraphDataset(data, client_train_indices)
    val_dataset = NodeSubgraphDataset(data, client_val_indices)
    
    # Store the full graph data for GNN message passing
    train_dataset.graph_data = data
    val_dataset.graph_data = data
    
    print(f"Client {client_id}: {len(train_dataset)} training nodes, "
          f"{len(val_dataset)} validation nodes")
    print(f"Client {client_id} class distribution:")
    train_labels = data.y[client_train_indices].numpy()
    for c in range(dataset.num_classes):
        count = (train_labels == c).sum()
        print(f"  Class {c}: {count} nodes ({100*count/len(train_labels):.1f}%)")
    
    return train_dataset, val_dataset
