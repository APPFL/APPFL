import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from collections import Counter
import random


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
    print(
        f"Balanced label distribution: {Counter(balanced_data_labels.view(-1).tolist())}"
    )

    # Recreate the dataset with balanced data
    balanced_train_dataset = [
        (balanced_data_input[i], balanced_data_labels[i])
        for i in range(len(balanced_data_labels))
    ]

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
        indices = np.random.choice(
            len(censored_inputs), len(event_inputs), replace=False
        )
        balanced_inputs = torch.cat((event_inputs, censored_inputs[indices]))
        balanced_labels = torch.cat((event_labels, censored_labels[indices]))
    else:
        indices = np.random.choice(
            len(event_inputs), len(censored_inputs), replace=False
        )
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
    pca_train_dataset = [
        (data_input_pca_tensor[i], label) for i, (_, label) in enumerate(train_dataset)
    ]

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
    scalers = [
        MinMaxScaler(feature_range=feature_range) for _ in range(data_input_np.shape[1])
    ]

    # Normalize each feature independently
    data_input_normalized_np = data_input_np.copy()
    for i in range(data_input_np.shape[1]):
        data_input_normalized_np[:, i] = (
            scalers[i].fit_transform(data_input_np[:, i].reshape(-1, 1)).flatten()
        )

    # Convert back to torch tensors
    data_input_normalized_tensor = torch.tensor(data_input_normalized_np)

    # Recreate the dataset with normalized data
    normalized_train_dataset = [
        (data_input_normalized_tensor[i], label)
        for i, (_, label) in enumerate(train_dataset)
    ]

    return normalized_train_dataset
