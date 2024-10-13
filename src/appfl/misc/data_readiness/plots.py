import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

from sklearn.decomposition import PCA
def plot_class_distribution(train_dataset):
    plt.figure()

    classes = train_dataset.data_label.unique().tolist()
    counts = []
    for c in classes:
        counts.append(train_dataset.data_label.tolist().count(c))

    plt.pie(counts, autopct='%1.1f%%')
    plt.legend(classes, loc='upper right')

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode the image as base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Close the buffer
    buffer.close()

    return encoded_image

def plot_data_sample(train_dataset):
    plt.figure(figsize=(15, 3))  

    data = train_dataset.data_input
    num_samples = 10
    total_samples = len(data)

    # Randomly select indices for the samples
    sample_indices = random.sample(range(total_samples), num_samples)

    # Create subplots for each sampled image
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)

        # Get the current sample
        sample = data[idx]

        # Determine the number of channels and image dimensions
        if sample.dim() == 2:  # For 2D tensors (H, W)
            channels = 1
            height, width = sample.shape
        elif sample.dim() == 3:  # For 3D tensors (C, H, W)
            channels, height, width = sample.shape
        else:
            raise ValueError(f"Unexpected tensor dimension: {sample.dim()}")

        # Reshape and display the image based on its dimensions
        if channels == 1:
            # If the image has 1 channel, use the 'gray' color map
            plt.imshow(sample.view(height, width).numpy(), cmap='gray', aspect='auto')
        elif channels == 3:
            # If the image has 3 channels, use the 'RGB' color map
            plt.imshow(sample.permute(1, 2, 0).numpy())
        else:
            raise ValueError(f"Unexpected number of channels: {channels}")

        plt.axis('off')

    # Adjust layout to prevent clipping
    plt.tight_layout(pad=1.0)

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free up memory

    # Encode the image as base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Close the buffer
    buffer.close()

    return encoded_image

def plot_data_distribution(train_dataset):
    # Flatten the pixel values
    data = random.sample(train_dataset.data_input.tolist(), 10)
    pixel_values = np.array(data).flatten()

    # Create a KDE plot of the data distribution
    plt.figure()
    sns.kdeplot(pixel_values)
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Calculate mean and standard deviation
    mean_value = np.mean(pixel_values)
    std_value = np.std(pixel_values)

    # Add mean and standard deviation to the plot
    plt.axvline(mean_value, color='r', linestyle='--', label='Mean')
    plt.axvline(mean_value + std_value, color='g', linestyle='--', label='Mean + Std')
    plt.axvline(mean_value - std_value, color='g', linestyle='--', label='Mean - Std')

    # Add legend to the plot
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode the image as base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Close the buffer
    buffer.close()

    return encoded_image

def plot_class_variance(train_dataset):
    # Get the unique classes in the dataset
    classes = torch.unique(train_dataset.data_label).tolist()

    # Create a dictionary to store the variance for each class
    class_variance = {}

    # Calculate the variance for each class
    for c in classes:
        # Create a boolean mask for the current class
        class_mask = train_dataset.data_label == c

        # Apply the mask to get the data for the current class
        class_data = train_dataset.data_input[class_mask]

        # Calculate the variance for the class
        variance_per_class = torch.var(class_data.float())  # Ensure the data is in float
        class_variance[c] = variance_per_class.item()  # Store as scalar value

    # Create a bar plot of the class variance
    plt.figure()
    plt.bar(class_variance.keys(), class_variance.values())
    plt.xlabel("Class")
    plt.ylabel("Variance")
    plt.title("Class-wise Variance")

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode the image as base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffer.close()

    return encoded_image

def get_feature_space_distribution(train_dataset):
    """
    Get feature space distribution for PCA visualization.
    
    Parameters:
    - train_dataset: The dataset from which to sample and calculate the PCA.

    Returns:
    - A dictionary containing PCA components, explained variance, and labels.
    """
    # Calculate 10% of the total dataset size for sampling
    total_samples = len(train_dataset.data_input)
    sample_size = max(1, int(total_samples * 0.1))  # Ensure at least one sample is taken
    
    # Randomly sample 10% of the data points from the feature space
    sampled_indices = random.sample(range(total_samples), sample_size)
    
    # Extract the sampled data
    data = [train_dataset.data_input[i] for i in sampled_indices]
    feature_values = np.array(data)  # Assuming this is of shape (batch_size, 1, 28, 28)

    # Reshape the 4D data (batch_size, 1, 28, 28) to 2D (batch_size, 28 * 28)
    batch_size = feature_values.shape[0]
    feature_values_flat = feature_values.reshape(batch_size, -1)  # Reshapes to (batch_size, 784)

    # Apply PCA to the flattened feature values (reduce to 2 components for visualization)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_values_flat)

    # Calculate the explained variance ratio for the PCA components
    explained_variance = pca.explained_variance_ratio_

    # Return the values needed to create the PCA plot on the frontend
    return {
        'pca_components': pca_result.tolist(),  # Convert PCA result to list for serialization
        'explained_variance': explained_variance.tolist(),  # Variance ratio for the components
        'labels': {
            'x_label': 'PCA Component 1',
            'y_label': 'PCA Component 2',
            'explained_variance_label': 'Explained Variance'
        }
    }


def generate_combined_feature_space_plot(client_feature_space_dict, client_ids):
    # Define a color palette for different clients
    colors = sns.color_palette("husl", len(client_ids))
    
    # Create a combined PCA plot for all clients' feature space
    plt.figure()

    for idx, (client_id, client_data) in enumerate(client_feature_space_dict.items()):
        pca_components = np.array(client_data['pca_components'])  # Get PCA components for each client

        # Ensure PCA components have the correct shape (2D: n_samples, 2)
        if pca_components.shape[1] != 2:
            raise ValueError(f"Expected PCA components with 2 dimensions, but got {pca_components.shape[1]} dimensions")
        
        # Apply jittering by adding small random noise to the points
        jitter_strength = 0.02
        jittered_components = pca_components + np.random.normal(scale=jitter_strength, size=pca_components.shape)
        
        # Plot the jittered PCA components for each client with reduced point size
        plt.scatter(jittered_components[:, 0], jittered_components[:, 1], 
                    color=colors[idx], label=client_id, alpha=0.4, s=20)

        # Optionally, display the explained variance for each client
        explained_variance = client_data['explained_variance']
        # plt.text(np.mean(jittered_components[:, 0]), np.mean(jittered_components[:, 1]),
        #         f'Var: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}',
        #         color=colors[idx])

    # Set labels for the PCA components
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Add legend to differentiate between clients
    plt.legend()

    # Save the plot to a BytesIO object and encode it as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Encode the plot as a base64 image to be returned
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image
