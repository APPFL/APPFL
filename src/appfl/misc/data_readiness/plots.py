import numpy as np
import torch
import random
import matplotlib
import seaborn as sns
import base64
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_segmentation_class_distribution(segmentation_map):
    """
    Plots the class distribution in a segmentation map.

    Args:
        segmentation_map (list, numpy.ndarray, or torch.Tensor): 
            A 2D or 3D segmentation map with pixel-wise class labels.

    Returns:
        str: Base64-encoded image of the class distribution pie chart.
    """
    # Convert list or other formats to a NumPy array
    if isinstance(segmentation_map, list):
        segmentation_map = np.array(segmentation_map)
    elif isinstance(segmentation_map, torch.Tensor):  # Handle PyTorch tensors
        segmentation_map = segmentation_map.cpu().numpy()
    
    # Flatten the segmentation map to count class occurrences
    flattened_map = segmentation_map.flatten()

    # Get unique classes and their counts
    classes, counts = np.unique(flattened_map, return_counts=True)

    # Plot the distribution
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
    plt.title("Segmentation Class Distribution")
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

    # Save the plot to a buffer as a base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

def plot_class_distribution(data_labels):
    
    plt.figure()

    classes = list(set(data_labels))
    counts = []
    for c in classes:
        counts.append(data_labels.count(c))

    plt.pie(counts, autopct='%1.1f%%')
    plt.legend(classes, loc='upper right')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffer.close()

    return encoded_image

def plot_data_sample(data_input):
    plt.figure(figsize=(15, 3))  

    data = data_input
    num_samples = 10
    total_samples = len(data)

    # Handle the case where total_samples is less than num_samples
    num_samples = min(num_samples, total_samples)
    sample_indices = random.sample(range(total_samples), num_samples)

    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)

        sample = data[idx]  # Extract a sample

        # Handle 2D Tensors (Height x Width)
        if sample.dim() == 2:
            height, width = sample.shape
            plt.imshow(sample.numpy(), cmap='gray', aspect='auto')

        # Handle 3D Tensors (Channels x Height x Width)
        elif sample.dim() == 3:
            channels, height, width = sample.shape

            if channels == 1:  # Grayscale
                plt.imshow(sample[0].numpy(), cmap='gray', aspect='auto')
            elif channels == 3:  # RGB
                plt.imshow(sample.permute(1, 2, 0).numpy())
            else:
                raise ValueError(f"Unexpected number of channels in 3D tensor: {channels}")

        # Handle 4D Tensors (1 x Depth x Height x Width)
        elif sample.dim() == 4:
            channels, depth, height, width = sample.shape

            if channels != 1:
                raise ValueError(f"Unexpected number of channels in 4D tensor: {channels}")
            
            # Select middle slice along the depth axis
            middle_slice = depth // 2
            slice_to_plot = sample[0, middle_slice, :, :]  # Shape: (Height x Width)
            plt.imshow(slice_to_plot.numpy(), cmap='gray', aspect='auto')

        # Handle 5D Tensors (Batch x Channels x Depth x Height x Width)
        elif sample.dim() == 5:
            batch, channels, depth, height, width = sample.shape

            if batch != 1:
                raise ValueError(f"Unexpected batch size in 5D tensor: {batch}")
            if channels != 1:
                raise ValueError(f"Unexpected number of channels in 5D tensor: {channels}")
            
            # Select middle slice along the depth axis of the first sample
            middle_slice = depth // 2
            slice_to_plot = sample[0, 0, middle_slice, :, :]  # Shape: (Height x Width)
            plt.imshow(slice_to_plot.numpy(), cmap='gray', aspect='auto')

        else:
            raise ValueError(f"Unexpected tensor dimension: {sample.dim()}")

        plt.axis('off')

    plt.tight_layout(pad=1.0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffer.close()

    return encoded_image

def plot_data_distribution(data_input):
    data = random.sample(data_input.tolist(), 10)
    pixel_values = np.array(data).flatten()

    plt.figure()
    sns.kdeplot(pixel_values)
    plt.xlabel("Value")
    plt.ylabel("Density")

    mean_value = np.mean(pixel_values)
    std_value = np.std(pixel_values)

    plt.axvline(mean_value, color='r', linestyle='--', label='Mean')
    plt.axvline(mean_value + std_value, color='g', linestyle='--', label='Mean + Std')
    plt.axvline(mean_value - std_value, color='g', linestyle='--', label='Mean - Std')

    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffer.close()

    return encoded_image

def plot_class_variance(data_input, data_labels):
    # Ensure data_labels is a tensor for indexing
    data_labels_tensor = torch.tensor(data_labels)
    classes = list(set(data_labels))

    class_variance = {}

    for c in classes:
        # Create a mask for the current class
        class_mask = data_labels_tensor == c

        # Select data for the current class
        class_data = data_input[class_mask]

        # Compute variance for the class data
        variance_per_class = torch.var(class_data.float())
        class_variance[c] = variance_per_class.item()

    # Plotting
    plt.figure()
    plt.bar(class_variance.keys(), class_variance.values())
    plt.xlabel("Class")
    plt.ylabel("Variance")
    plt.title("Class-wise Variance")

    # Save plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffer.close()

    return encoded_image

def get_feature_space_distribution(data_input):
    # Get the shape of the input data
    shape = data_input.shape
    
    # Flatten the data into a 2D array if it has 2 or more dimensions, otherwise reshape it to 2D
    if len(shape) >= 2:
        feature_dim = np.prod(shape[1:])
        feature_values = data_input.reshape(-1, feature_dim)
    else:
        feature_values = data_input.reshape(-1, 1)

    # Compute PCA with 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feature_values)

    # Compute variance explained by each PCA component
    explained_variance = pca.explained_variance_ratio_

    # Return the results as a dictionary
    return {
        'pca_components': pca_result.tolist(),
        'explained_variance': explained_variance.tolist(),
        'labels': {
            'x_label': 'PCA Component 1',
            'y_label': 'PCA Component 2',
            'z_label': 'PCA Component 3',
            'explained_variance_label': 'Explained Variance'
        }
    }

def generate_combined_feature_space_plot(client_feature_space_dict, client_ids, sample_size=100):
    colors = sns.color_palette("husl", len(client_ids))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    jitter_strength = 0.05

    for idx, (client_id, client_data) in enumerate(client_feature_space_dict.items()):
        pca_components = np.array(client_data['pca_components'])
        
        if pca_components.shape[1] != 3:
            raise ValueError(f"Expected PCA components with 3 dimensions, but got {pca_components.shape[1]} dimensions")
        
        # Sample points to ensure each client has the same number represented
        if pca_components.shape[0] > sample_size:
            sampled_indices = random.sample(range(pca_components.shape[0]), sample_size)
            pca_components = pca_components[sampled_indices, :]
        
        jittered_components = pca_components + np.random.normal(scale=jitter_strength, size=pca_components.shape)
        
        ax.scatter(jittered_components[:, 0], jittered_components[:, 1], jittered_components[:, 2],
                   color=colors[idx], label=client_id, alpha=0.6, s=30)
        
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")

    ax.legend(loc='upper right')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

def plot_outliers(data_input):
    num_features = data_input.shape[1]
    outlier_proportions = []

    for feature_idx in range(num_features):
        # Extract feature values
        feature_values = data_input[:, feature_idx].numpy()
        
        # Calculate Q1, Q3, and IQR
        q1 = np.percentile(feature_values, 25)
        q3 = np.percentile(feature_values, 75)
        iqr = q3 - q1

        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = (feature_values < lower_bound) | (feature_values > upper_bound)

        # Calculate the proportion of outliers
        outlier_proportion = np.sum(outliers) / len(feature_values)
        
        # Consider only features with outliers
        if outlier_proportion > 0:
            outlier_proportions.append(outlier_proportion)

    # Create bar chart
    plt.figure()
    sns.barplot(x=[f'{i + 1}' for i in range(len(outlier_proportions))], y=outlier_proportions, palette='viridis')
    plt.title('Proportion of Outliers in Features with Outliers')
    plt.ylabel('Proportion of Outliers')
    plt.xlabel('Features')
    plt.xticks(rotation=45)

    # Save plot to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

def plot_representative_rates(data_input, fairness_feature_idx):
    if fairness_feature_idx != None:
        # Extract fairness feature values
        fairness_values = data_input[:, fairness_feature_idx].numpy()

        # Calculate the proportion of each unique value
        unique_values, value_counts = np.unique(fairness_values, return_counts=True)
        value_proportions = value_counts / len(fairness_values)

        # Create bar chart
        plt.figure()
        sns.barplot(x=unique_values, y=value_proportions, palette='viridis')
        plt.title('Proportion of Each Unique Value in the Fairness Feature')
        plt.ylabel('Proportion')
        plt.xlabel('Unique Values')
        plt.xticks(rotation=45)

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        # Encode image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        return encoded_image

def plot_feature_correlations(data_input):

    # Convert data to numpy array
    data_array = data_input.numpy()

    # Compute correlation matrix
    correlation_matrix = np.corrcoef(data_array, rowvar=False)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 10))

    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={"size": 12},  # Annotation font size (inside cells)
                xticklabels=16,  # x-axis tick label font size
                yticklabels=16  # y-axis tick label font size
                )

    plt.xticks(fontsize=16)  # Font size of x-axis ticks
    plt.yticks(fontsize=16)  # Font size of y-axis ticks


    # Save to buffer and encode
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

def plot_feature_importance(data_input, data_labels):
    # Convert data to numpy array
    data_array = data_input.numpy()

    # Compute feature importance using a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(data_array, data_labels)
    importances = clf.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances
    plt.figure()
    plt.bar(range(data_array.shape[1]), importances[indices], align='center')
    plt.xticks(range(data_array.shape[1]), indices, rotation=90)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')

    # Save to buffer and encode
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

def plot_incompleteness(data_input):
    # Convert data_input to numpy array
    data_array = data_input.numpy()

    # Calculate the proportion of missing values in each feature
    nan_mask = np.isnan(data_array)
    nan_mask_float = nan_mask.astype(float)
    missing_proportions = np.mean(nan_mask_float, axis=0)

    # Check if all features are complete
    if np.all(missing_proportions == 0):
        return "All features are complete."

    # Create bar chart
    plt.figure()
    sns.barplot(x=[f'{i+1}' for i in range(len(missing_proportions))], y=missing_proportions, palette='viridis')
    plt.title('Proportion of Missing Values in Each Feature')
    plt.ylabel('Proportion of Missing Values')
    plt.xlabel('Features')
    plt.xticks(rotation=45)

    # Save plot to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image
    

def calculate_entropy(data):
    """Calculate entropy of a 1D PyTorch tensor feature."""
    hist, _ = np.histogram(data.numpy(), bins=10, density=True)
    hist = hist + 1e-12  # Avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def plot_feature_statistics(data_input):
    """
    Generates a radar chart of feature statistics including sparsity, outlier proportion,
    skewness, variance, kurtosis, mean, standard deviation, and entropy,
    then returns the plot as a base64-encoded image.

    Parameters:
    data_input (torch.Tensor): A PyTorch tensor where each column represents a feature.

    Returns:
    str: A base64-encoded image representing the radar chart.
    """
    # Initialize lists to hold statistic values for each feature
    sparsity = []
    outlier_proportion = []
    skewness = []
    variance = []
    kurtosis_values = []
    mean = []
    std_dev = []
    entropy = []

    # Calculate statistics for each feature (column)
    for i in range(data_input.shape[1]):
        feature_data = data_input[:, i]
        
        # Sparsity (proportion of zeros)
        sparsity.append((feature_data == 0).float().mean().item())
        
        # Outlier proportion using IQR
        q1 = torch.quantile(feature_data, 0.25)
        q3 = torch.quantile(feature_data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
        outlier_proportion.append(outliers.float().mean().item())
        
        # Skewness
        feature_mean = feature_data.mean()
        feature_std = feature_data.std()
        if feature_std == 0:
            skewness.append(0)  # If std is zero, skewness is zero by definition.
        else:
            skewness.append(((feature_data - feature_mean) ** 3).mean().item() / (feature_std ** 3))
        
        # Variance
        variance.append(feature_data.var(unbiased=False).item())
        
        # Kurtosis
        if feature_std == 0:
            kurtosis_values.append(-3)  # A constant feature has excess kurtosis of -3
        else:
            kurtosis_values.append(((feature_data - feature_mean) ** 4).mean().item() / (feature_std ** 4) - 3)
        
        # Mean
        mean.append(feature_mean.item())
        
        # Standard Deviation
        std_dev.append(feature_std.item())
        
        # Entropy
        entropy.append(calculate_entropy(feature_data))

    # Custom normalization for specific metrics
    def normalize(lst, min_val=None, max_val=None):
        lst = np.array(lst)
        if min_val is not None and max_val is not None:
            return (lst - min_val) / (max_val - min_val)
        if np.max(lst) == np.min(lst):
            return np.zeros_like(lst)  # If all values are the same, return zeros.
        return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))

    # Normalizing specific metrics, handling edge cases
    normalized_outliers = outlier_proportion
    normalized_variance = normalize(variance)
    normalized_kurtosis = normalize(kurtosis_values)
    normalized_mean = normalize(mean)
    normalized_std_dev = normalize(std_dev)
    normalized_entropy = normalize(entropy)

    # Sparsity doesn't need normalization (it's already between 0 and 1)
    sparsity = np.array(sparsity)

    # Skewness can be very negative or very positive, normalize it to a fixed range (-10 to 10 for example)
    skewness = np.clip(np.array(skewness), -10, 10)  # Limiting skewness for extreme values
    normalized_skewness = normalize(skewness, -10, 10)

    # Prepare data for radar chart
    statistics = list(zip(
        sparsity, normalized_outliers, normalized_skewness, normalized_variance
    ))
    categories = ['Sparsity', 'Outliers', 'Skewness', 'Variance']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Add each feature's statistics to the plot
    for i, stats in enumerate(statistics):
        # Close the radar chart loop
        stats_extended = np.concatenate((stats, [stats[0]]))
        angles_extended = np.concatenate((angles, [angles[0]]))
        ax.plot(angles_extended, stats_extended, linewidth=1, linestyle='solid', label=f'Feature {i}')
        ax.fill(angles_extended, stats_extended, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=12)

    # Add a legend outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)


    # Save plot to a buffer and encode as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_str

def plot_metric_pca(metrics_data,output_dir="output", filename="pca_plot.png"):
    """
    Generate and save a PCA plot for client metrics.

    Args:
        metrics_data (dict): Dictionary where keys are client IDs and values are 
                             dictionaries of metrics. Example:
                             {
                                 1: {'metric1': value1, 'metric2': value2, ...},
                                 2: {'metric1': value1, 'metric2': value2, ...},
                                 ...
                             }
        output_dir (str): Directory where the plot will be saved. Default is "output".
        filename (str): Name of the file to save the plot as. Default is "pca_plot.png".
    """
    # Extract client IDs and metrics
    ids = list(metrics_data.keys())
    metrics = list(metrics_data[ids[0]].keys())
    data = np.array([[metrics_data[id_][metric] for metric in metrics] for id_ in ids])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(pca_result):
        plt.scatter(x, y, label=f"Client {ids[i]}")
        plt.text(x + 0.02, y + 0.02, f"{ids[i]}", fontsize=9)

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.title("PCA Plot of Client Metrics")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Clients")
    plt.grid(True)

    # Save the plot to the specified output directory
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    print(f"PCA plot saved at: {file_path}")