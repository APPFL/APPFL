import numpy as np
import torch
import piq
from typing import Dict


def imbalance_degree(lst):
    # Count occurrences of each class
    counts = {}
    for elem in lst:
        counts[elem] = counts.get(elem, 0) + 1

    # Check if only one class exists
    num_classes = len(counts)
    if num_classes == 1:
        # Handle the case where there's only one class
        return float(
            "inf"
        )  # Or return a specific value like 0, or None, or raise an error

    # Calculate imbalance degree
    total_elements = len(lst)
    actual_proportions = np.array(
        [counts[elem] / total_elements for elem in sorted(counts.keys())]
    )
    balanced_proportions = np.array([1 / num_classes] * num_classes)
    euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)

    return euclidean_distance


def completeness(data):
    num_non_nan = torch.sum(~torch.isnan(data))
    total_elements = data.numel()
    completeness = num_non_nan.item() / total_elements
    rounded_completeness = round(completeness, 2)
    return rounded_completeness


def sparsity(data):
    total_elements = data.numel()
    num_zeros = torch.sum(data == 0)
    sparsity = num_zeros.item() / total_elements
    rounded_sparsity = round(sparsity, 2)
    return rounded_sparsity


def variance(data):
    variance = torch.var(data)
    rounded_variance = round(variance.item(), 2)
    return rounded_variance


def skewness(data):
    data = data.numpy().flatten()
    mean = np.mean(data)
    std_dev = np.std(data)
    median = np.median(data)
    skewness = (3 * (mean - median)) / std_dev
    rounded_skewness = round(skewness, 2)
    return rounded_skewness


def entropy(data):
    data = data.numpy().flatten()
    hist = np.histogram(data, bins=256, range=(0, 1), density=True)[0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    rounded_entropy = round(entropy, 2)
    return rounded_entropy


def kurtosis(data):
    data = data.numpy().flatten()
    mean = np.mean(data)
    std_dev = np.std(data)
    kurtosis = np.sum((data - mean) ** 4) / (len(data) * std_dev**4)
    rounded_kurtosis = round(kurtosis, 2)
    return rounded_kurtosis


def class_distribution(labels):
    counts = {}
    for elem in labels:
        counts[elem] = counts.get(elem, 0) + 1
    return counts


def get_data_range(data):
    if data.numel() == 0:
        print("Warning: The data tensor is empty.")
        return {"min": None, "max": None}
    if torch.isnan(data).any():
        print(
            "Warning: The data contains NaN values. Removing NaN for min/max calculation."
        )
        data = data[~torch.isnan(data)]
    if data.numel() == 0:
        print("Warning: All data values are NaN.")
        return {"min": None, "max": None}
    data = data.float()
    data_min = torch.min(data).item()
    data_max = torch.max(data).item()
    if np.isnan(data_min) or np.isnan(data_max):
        print("Warning: Min or Max is NaN after calculation.")
        return {"min": None, "max": None}
    return {"min": round(data_min, 2), "max": round(data_max, 2)}


def brisque(data):
    brisque_score = piq.brisque(data)
    return round(brisque_score.item(), 2)


def total_variation(data):
    tv = piq.total_variation(data)
    return round(tv.item(), 2)


def image_sharpness(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = np.mean(image, axis=0)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    conv = np.abs(np.convolve(image.flatten(), laplacian.flatten(), mode="same"))
    var = np.var(conv)
    return var


def dataset_sharpness(dataset):
    sharpness_scores = [image_sharpness(img) for img in dataset]
    avg_sharpness = np.mean(sharpness_scores)
    return round(avg_sharpness, 2)


def ned_squared(
    class_distribution1: Dict[int, int], class_distribution2: Dict[int, int]
) -> float:
    u = np.array(list(class_distribution1.values()))
    v = np.array(list(class_distribution2.values()))
    var_u = np.var(u)
    var_v = np.var(v)
    var_diff = np.var(u - v)
    if (var_u + var_v) == 0:
        return 0
    return 0.5 * var_diff / (var_u + var_v)


def calculate_outlier_proportion(data_input):
    # Convert to NumPy if data_input is a PyTorch tensor
    if isinstance(data_input, torch.Tensor):
        data_input = data_input.numpy()

    num_features = data_input.shape[1]
    total_outliers = 0
    total_data_points = data_input.shape[0] * num_features

    for feature_idx in range(num_features):
        feature_values = data_input[:, feature_idx]

        q1 = np.percentile(feature_values, 25)
        q3 = np.percentile(feature_values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = np.sum(
            (feature_values < lower_bound) | (feature_values > upper_bound)
        )
        total_outliers += outlier_count

    proportion_outliers = total_outliers / total_data_points

    return round(proportion_outliers, 2)


def quantify_time_to_event_imbalance(data_label):
    # Convert list of tensors to numpy array
    data_array = np.array([tensor.numpy() for tensor in data_label])

    # Split data into time and event status
    T = data_array[:, 0]  # Event observed (1) or censorship (0)
    E = data_array[:, 1]  # Relative risk

    # Calculate proportions
    n_total = len(T)
    n_events = np.sum(T == 1)
    n_censored = np.sum(T == 0)

    event_proportion = n_events / n_total
    censored_proportion = n_censored / n_total

    # Calculate average risk
    E_event_avg = np.mean(E[T == 1]) if n_events else 0
    E_censored_avg = np.mean(E[T == 0]) if n_censored else 0
    E_max = np.max(E) if len(E) else 1

    # Calculate normalized differences
    proportion_imbalance = abs(event_proportion - censored_proportion)
    risk_imbalance = abs(E_event_avg - E_censored_avg) / E_max

    # Calculate Combined Imbalance Score
    combined_imbalance_score = (proportion_imbalance + risk_imbalance) / 2

    return round(combined_imbalance_score, 2)
