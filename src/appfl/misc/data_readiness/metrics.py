import numpy as np
import torch
import piq

def imbalance_degree(lst):
    counts = {}
    for elem in lst:
        counts[elem] = counts.get(elem, 0) + 1
    total_elements = len(lst)
    actual_proportions = np.array([counts[elem] / total_elements for elem in sorted(counts.keys())])
    num_classes = len(counts)
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
    kurtosis = np.sum((data - mean) ** 4) / (len(data) * std_dev ** 4)
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
        print("Warning: The data contains NaN values. Removing NaN for min/max calculation.")
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
    return {"min": data_min, "max": data_max}

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
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    conv = np.abs(np.convolve(image.flatten(), laplacian.flatten(), mode='same'))
    var = np.var(conv)
    return var

def dataset_sharpness(dataset):
    sharpness_scores = [image_sharpness(img) for img in dataset]
    avg_sharpness = np.mean(sharpness_scores)
    return round(avg_sharpness, 2)


