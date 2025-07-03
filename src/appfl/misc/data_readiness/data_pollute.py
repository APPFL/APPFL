import random
import torch


def add_noise_to_subset(dataset, scale, fraction, seed=42):
    """
    Add random noise to a fraction of the input data in the dataset.

    Parameters:
    - dataset: Dataset where each item is a tuple (input_data, label).
    - scale: Scale of the Gaussian noise.
    - fraction: Fraction of data to add noise to (0 to 1).

    Returns:
    - modified_dataset: List with partially noisy input data and original labels.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
        noisy_input = (input_data + noise).clamp(2, 5)
        dataset_list[idx] = (noisy_input, label)

    return dataset_list


def add_duplicates(dataset, duplicate_ratio):
    """
    Add a proportion of duplicate samples to the dataset.

    Parameters:
    - dataset: Dataset where each item is a tuple (input_data, label).
    - duplicate_ratio: Float between 0 and 1 indicating the proportion of duplicates
                       to add (e.g., 0.2 means 20% duplicates of the original size).

    Returns:
    - modified_dataset: List with added duplicate samples.
    """
    if not (0 <= duplicate_ratio <= 1):
        raise ValueError("duplicate_ratio must be between 0 and 1.")

    dataset_list = list(dataset)
    num_duplicates = int(len(dataset_list) * duplicate_ratio)

    # Randomly select samples (with replacement) to duplicate
    duplicate_indices = random.choices(range(len(dataset_list)), k=num_duplicates)

    # Add duplicates
    for idx in duplicate_indices:
        input_data, label = dataset_list[idx]
        dataset_list.append((input_data, label))

    return dataset_list
