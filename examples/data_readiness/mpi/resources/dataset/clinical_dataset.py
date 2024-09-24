import os
import pandas as pd
import numpy as np
import torch
from appfl.misc.data import Dataset, iid_partition, class_noniid_partition, dirichlet_noniid_partition

def get_clinical(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    **kwargs
):
    # Read CSV file
    csv_path = os.path.join(os.getcwd(), "datasets", "RawData", "clinical_data.csv")
    data = pd.read_csv(csv_path)

    print(f"Original data shape: {data.shape}")

    # Handle each column based on its content
    for column in data.columns:
        if data[column].dtype == 'object':
            # Try to convert to numeric, if fails, encode as categorical
            try:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            except:
                data[column] = pd.Categorical(data[column]).codes
        elif data[column].dtype == 'bool':
            data[column] = data[column].astype(int)

    # Handle NaN values
    data = data.fillna(data.mean())

    # Ensure all data is numeric
    numeric_data = data.select_dtypes(include=[np.number])

    # Convert to numpy array
    features = numeric_data.values.astype(np.float32)

    # Assume the last column is the label
    labels = features[:, -1].astype(np.int64)
    features = features[:, :-1]

    print(f"Processed features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)

    # Create a Dataset using the APPFL Dataset class
    full_dataset = Dataset(features_tensor, labels_tensor)

    # Partition the dataset based on the strategy
    if partition_strategy == "iid":
        train_datasets = iid_partition(full_dataset, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(full_dataset, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(full_dataset, num_clients, **kwargs)
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    # Get the dataset for the specific client
    train_dataset = train_datasets[client_id]

    # For simplicity, assume the test set is the entire dataset (adjust as needed)
    test_dataset = full_dataset

    return train_dataset, test_dataset