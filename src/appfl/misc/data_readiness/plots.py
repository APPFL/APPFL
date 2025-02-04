import numpy as np
import torch
import random
import matplotlib
import seaborn as sns
import base64
import io
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter


matplotlib.use('Agg')

import matplotlib.pyplot as plt

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

    sample_indices = random.sample(range(total_samples), num_samples)

    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)

        sample = data[idx]

        if sample.dim() == 2:
            channels = 1
            height, width = sample.shape
        elif sample.dim() == 3:
            channels, height, width = sample.shape
        else:
            raise ValueError(f"Unexpected tensor dimension: {sample.dim()}")

        if channels == 1:
            plt.imshow(sample.view(height, width).numpy(), cmap='gray', aspect='auto')
        elif channels == 3:
            plt.imshow(sample.permute(1, 2, 0).numpy())
        else:
            raise ValueError(f"Unexpected number of channels: {channels}")

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
    
    plt.figure(figsize=(20, 10))
    
    for feature_idx in range(num_features):
        # Extract feature values
        feature_values = data_input[:, feature_idx].numpy()
        
        # Create subplot for each feature
        plt.subplot(2, 7, feature_idx + 1)  # 2 rows, 7 columns layout
        sns.boxplot(x=feature_values)
        plt.title(f'Feature {feature_idx + 1}')
        plt.xlabel("Value")
    
    plt.tight_layout()
    
    # Save plot to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    
    # Encode image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    
    return encoded_image

def plot_time_to_event_distribution(data_label):
    # Convert list of tensors to numpy array
    data_array = np.array([tensor.numpy() for tensor in data_label])

    # Split data into event indicator and time
    E = data_array[:, 0]  # Event occurred
    T = data_array[:, 1]  # Time to event or censoring

    # Instantiate KaplanMeierFitter
    kmf = KaplanMeierFitter()

    # Fit the data
    kmf.fit(durations=T, event_observed=E)

    # Plot the survival function
    plt.figure(figsize=(10, 5))
    kmf.plot_survival_function()
    plt.title('Kaplan-Meier Survival Estimate')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')

    # Save to buffer and encode
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image
