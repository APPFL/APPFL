import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import numpy as np
from collections import Counter
import random
from sklearn.cluster import KMeans
import pandas as pd
import os
import matplotlib.pyplot as plt

def balance_classes_undersample(train_dataset):
    """
    Manually balance the dataset using undersampling to reduce majority class samples.

    Parameters:
    - train_dataset: Dataset where each item is a tuple (input_data, label).

    Returns:
    - balanced_train_dataset: List with balanced input data and labels.
    """
    # Separate inputs and labels
    data_input = [input_data for input_data, _ in train_dataset]
    data_labels = [label.item() for _, label in train_dataset]

    # Count label occurrences
    label_counter = Counter(data_labels)
    print(f"Original label distribution: {label_counter}")

    # Identify the minority class size
    minority_class_size = min(label_counter.values())

    # Group data by label
    label_to_data = {}
    for input_data, label in zip(data_input, data_labels):
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append((input_data, label))

    # Undersample to match the minority class size
    balanced_data = []
    for label, items in label_to_data.items():
        sampled_items = random.sample(items, minority_class_size)
        balanced_data.extend(sampled_items)

    # Shuffle the balanced dataset
    random.shuffle(balanced_data)

    # Ensure consistent shapes with view
    balanced_data_input = torch.stack([item[0] for item in balanced_data])
    balanced_data_labels = torch.tensor([item[1] for item in balanced_data]).view(-1, 1)

    # Print balanced dataset distribution
    print(f"Balanced label distribution: {Counter(balanced_data_labels.view(-1).tolist())}")

    # Recreate the dataset with balanced data
    balanced_train_dataset = [(balanced_data_input[i], balanced_data_labels[i]) for i in range(len(balanced_data_labels))]

    return balanced_train_dataset

def balance_data(data_input, data_label):
    # Ensure data_input and data_label are tensors
    if isinstance(data_input, list):
        data_input = torch.stack(data_input)

    if isinstance(data_label, list):
        data_label = torch.stack(data_label)

    # Separate into events and censored
    event_mask = data_label[:, 0] == 1  # Assuming first column is the event indicator
    censored_mask = data_label[:, 0] == 0

    event_inputs = data_input[event_mask]
    censored_inputs = data_input[censored_mask]

    event_labels = data_label[event_mask]
    censored_labels = data_label[censored_mask]

    # Balancing logic: undersample the majority class
    if len(event_inputs) < len(censored_inputs):
        indices = np.random.choice(len(censored_inputs), len(event_inputs), replace=False)
        balanced_inputs = torch.cat((event_inputs, censored_inputs[indices]))
        balanced_labels = torch.cat((event_labels, censored_labels[indices]))
    else:
        indices = np.random.choice(len(event_inputs), len(censored_inputs), replace=False)
        balanced_inputs = torch.cat((event_inputs[indices], censored_inputs))
        balanced_labels = torch.cat((event_labels[indices], censored_labels))

    return balanced_inputs, balanced_labels


def apply_pca_to_dataset(train_dataset, n_components=30):
    """
    Apply PCA to reduce dimensionality of the input data in train_dataset.

    Parameters:
    - train_dataset: Dataset where each item is a tuple (input_data, label).
    - n_components: Number of principal components to keep.

    Returns:
    - pca_train_dataset: List with PCA-transformed input data and original labels.
    """
    # Stack the input data from the dataset
    data_input = torch.stack([input_data for input_data, _ in train_dataset])
    
    # Convert data_input to numpy array
    data_input_np = data_input.numpy()

    # Standardize the data_input
    scaler = StandardScaler()
    data_input_scaled = scaler.fit_transform(data_input_np)

    # Apply PCA
    if n_components is None:
        n_components = min(data_input_np.shape[0], data_input_np.shape[1])

    pca = PCA(n_components=n_components)
    data_input_pca = pca.fit_transform(data_input_scaled)

    # Convert back to torch tensors
    data_input_pca_tensor = torch.tensor(data_input_pca)

    # Recreate the dataset with PCA-transformed data
    pca_train_dataset = [(data_input_pca_tensor[i], label) for i, (_, label) in enumerate(train_dataset)]

    return pca_train_dataset

def normalize_dataset(train_dataset, feature_range=(0, 1)):
    """
    Normalize each feature of the input data from train_dataset to a specified range.

    Parameters:
    - train_dataset: Dataset where each item is a tuple (input_data, label).
    - feature_range: Tuple specifying the desired range (min, max).

    Returns:
    - normalized_train_dataset: List with normalized input data and original labels.
    """
    # Stack the input data from the dataset into a single tensor
    data_input = torch.stack([input_data for input_data, _ in train_dataset])

    # Convert data_input to a numpy array for normalization
    data_input_np = data_input.numpy()

    # Initialize the MinMaxScaler for each feature (column)
    scalers = [MinMaxScaler(feature_range=feature_range) for _ in range(data_input_np.shape[1])]

    # Normalize each feature independently
    data_input_normalized_np = data_input_np.copy()
    for i in range(data_input_np.shape[1]):
        data_input_normalized_np[:, i] = scalers[i].fit_transform(data_input_np[:, i].reshape(-1, 1)).flatten()

    # Convert back to torch tensors
    data_input_normalized_tensor = torch.tensor(data_input_normalized_np)

    # Recreate the dataset with normalized data
    normalized_train_dataset = [(data_input_normalized_tensor[i], label) for i, (_, label) in enumerate(train_dataset)]

    return normalized_train_dataset

def find_most_diverse_client(metric_mappings, n_clusters=1, output_dir="output", output_filename="kmeans_plot.png"):
    """
    Determines the most diverse client based on metric mappings using K-Means clustering and saves a plot of the clusters.

    Parameters:
    - metric_mappings (dict): A dictionary where keys are client IDs and values are dictionaries of metrics.
    - n_clusters (int): The number of clusters to form and the number of centroids to generate.
    - output_dir (str): The directory where the plot will be saved.
    - output_filename (str): The name of the file to save the plot as.

    Returns:
    - int: The client ID of the most diverse client.
    """
    # Convert the metric mappings to a DataFrame
    data = pd.DataFrame(metric_mappings).T
    client_ids = list(metric_mappings.keys())

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data)

    # Calculate the distance to cluster centers for each client
    data['distance_to_center'] = np.linalg.norm(
        data.iloc[:, :-1] - kmeans.cluster_centers_[data['cluster']], axis=1
    )

    # Find the client with the maximum distance to the cluster center
    most_diverse_client = data['distance_to_center'].idxmax()

    # Perform PCA to reduce data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data.iloc[:, :-2])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['cluster'], cmap='viridis', marker='o', alpha=0.7)

    for i, client_id in enumerate(client_ids):
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(client_id), fontsize=8, ha='right')

    
    # Mark the centroids
    pca_centroids = pca.transform(kmeans.cluster_centers_)
    plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    

    plt.title('K-Means Clustering of Clients')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # Save plot to the specified directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()

    return data['distance_to_center']

def compute_client_weights(distances, min_weight=0.01):
    """
    Normalize distances and compute weights for clients, ensuring weights add up to 1.

    Args:
        distances (dict): A dictionary where keys are client IDs and values are distances.
                          Example: {1: 2.614750, 2: 1.026734, ...}
        min_weight (float): Minimum weight value to ensure weights are not zero. Default is 0.01.

    Returns:
        dict: A dictionary where keys are client IDs and values are the computed weights, summing to 1.
    """
    # Ensure distances is a dictionary
    distances = dict(distances)

    # Extract distance values and compute min-max normalization
    dist_values = np.array(list(distances.values()))
    min_dist = dist_values.min()
    max_dist = dist_values.max()
    normalized_distances = (dist_values - min_dist) / (max_dist - min_dist)

    # Compute base weights (higher distance -> lower weight)
    base_weights = 1 - normalized_distances

    # Scale weights to ensure no weight is zero
    weight_range = 1 - min_weight
    scaled_weights = weight_range * base_weights + min_weight

    # Normalize weights to ensure they sum to 1
    normalized_weights = scaled_weights / scaled_weights.sum()

    # Map weights back to client IDs
    client_weights = {client: weight for client, weight in zip(distances.keys(), normalized_weights)}

    return client_weights

def add_noise_to_subset(dataset, scale, fraction):
    """
    Add random noise to a fraction of the input data in the dataset.

    Parameters:
    - dataset: Dataset where each item is a tuple (input_data, label).
    - scale: Scale of the Gaussian noise.
    - fraction: Fraction of data to add noise to (0 to 1).

    Returns:
    - modified_dataset: List with partially noisy input data and original labels.
    """
    # Convert dataset to list for easy manipulation
    dataset_list = list(dataset)
    
    # Determine number of samples to add noise to
    num_noisy_samples = int(len(dataset_list) * fraction)
    
    # Randomly select indices for noisy samples
    noisy_indices = random.sample(range(len(dataset_list)), num_noisy_samples)
    
    # Add noise to selected samples
    for idx in noisy_indices:
        input_data, label = dataset_list[idx]
        noise = torch.randn_like(input_data) * scale
        noisy_input = (input_data + noise).clamp(0, 1)
        dataset_list[idx] = (noisy_input, label)
    
    return dataset_list

def sample_subset(dataset, sample_size):
    """
    Sample a subset of the dataset.
    
    :param dataset: the dataset to sample from (should be a PyTorch Dataset or list)
    :param sample_size: the size of the sample
    :return: a subset of the dataset
    """
    if len(dataset) <= sample_size:
        return dataset
    else:
        # Randomly permute indices and select a subset
        indices = torch.randperm(len(dataset))[:sample_size]
        # Use list comprehension to create the subset
        sampled_data = [dataset[i] for i in indices]
        return sampled_data