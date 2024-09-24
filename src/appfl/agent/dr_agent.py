import numpy as np
from appfl.agent import ClientAgent
import torch
import piq
import matplotlib.pyplot as plt
import base64
import io
import random
import seaborn as sns

class DRAgent(ClientAgent):
    """
    
    Developed by Kaveen Hiniduma(hiniduma.1@osu.edu) and Suren Byna(byna.1@osu.edu) from The Ohio State University.

    This class is used to generate data readiness reports for a client agent.
    The data readiness report includes standard metrics and plots that provide insights into the data distribution and quality.
    The class inherits from the ClientAgent class and extends it with data readiness functionality.

    """

    def generate_readiness_report(self):
        if hasattr(self.client_agent_config, "dr_metrics"):
            
            results = {}
            plot_results = {"plots": {}}

            # Define metrics with corresponding computation functions
            standard_metrics = {
                "dataset_name": self._get_dataset_name,  
                "class_imbalance": lambda: round(self._imbalance_degree(self.train_dataset.data_label.tolist()), 2),
                "sample_size": lambda: len(self.train_dataset.data_label.tolist()),
                "num_classes": lambda: len(self.train_dataset.data_label.unique()),
                "data_shape": lambda: list(self.train_dataset.data_input.shape),
                "completeness": self._completeness,
                "data_range": self._get_data_range,
                "sparsity": self._sparsity,
                "variance": self._variance,
                "skewness": self._skewness,
                "entropy": self._entropy,
                "kurtosis": self._kurtois,
                "class_distribution": self._class_distribution,
                "brisque": self._brisque,
                "sharpness": self._dataset_sharpness,
                "total_variation": self._total_variation,
            }

            plots = {
                "class_distribution_plot": self._plot_class_distribution,
                "data_sample_plot": self._plot_data_sample,
                "data_distribution_plot": self._plot_data_distribution,
                "class_variance_plot": self._plot_class_variance,
           
            }

            # First loop: Handle standard metrics
            for metric_name, compute_function in standard_metrics.items():
                if metric_name in self.client_agent_config.dr_metrics:
                    if getattr(self.client_agent_config.dr_metrics, metric_name):
                        results[metric_name] = compute_function()
                    else:
                        results[metric_name] = "Metric set to False during configuration"
                else:
                    results[metric_name] = "Metric not available in configuration"

            # Second loop: Handle plot-specific metrics
            for metric_name, compute_function in plots.items():
                if metric_name in self.client_agent_config.dr_metrics.plot:
                    if getattr(self.client_agent_config.dr_metrics.plot, metric_name):
                        plot_results['plots'][metric_name] = compute_function()
                    else:
                        plot_results['plots'][metric_name] = "Plot metric set to False during configuration"
                else:
                    plot_results['plots'][metric_name] = "Plot metric not available in configuration"

            # Combine results with plot results
            results.update(plot_results)

            return results
        
        else:
            return "Data readiness metrics not available in configuration"
        
    def _get_dataset_name(self):
        return self.client_agent_config.data_configs.dataset_name
        
    def _imbalance_degree(self, lst):
        """
        Calculate the proportions of each class and the proportions for a balanced distribution and return the euclidean distance between the two distributions.
        """
        counts = {}
        for elem in lst:
            counts[elem] = counts.get(elem, 0) + 1
        total_elements = len(lst)
        actual_proportions = np.array([counts[elem] / total_elements for elem in sorted(counts.keys())])
        num_classes = len(counts)
        balanced_proportions = np.array([1 / num_classes] * num_classes)
        euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)
        return euclidean_distance
    
    def _completeness(self):

        data = self.train_dataset.data_input
        num_non_nan = torch.sum(~torch.isnan(data))
        total_elements = data.numel()
        completeness = num_non_nan.item() / total_elements
        rounded_completeness = round(completeness, 2)
        return rounded_completeness
    
    def _sparsity(self):
        data = self.train_dataset.data_input
        total_elements = data.numel()
        num_zeros = torch.sum(data == 0)
        sparsity = num_zeros.item() / total_elements
        rounded_sparsity = round(sparsity, 2)
        return rounded_sparsity
    
    def _variance(self):
        data = self.train_dataset.data_input
        variance = torch.var(data)
        rounded_variance = round(variance.item(), 2)
        return rounded_variance
    
    def _skewness(self):
        data = self.train_dataset.data_input.numpy().flatten()
        mean = np.mean(data)
        std_dev = np.std(data)
        median = np.median(data)
        skewness = (3 * (mean - median)) / std_dev
        rounded_skewness = round(skewness, 2)
        return rounded_skewness
    
    def _entropy(self):
        data = self.train_dataset.data_input.numpy().flatten()
        hist = np.histogram(data, bins=256, range=(0, 1), density=True)[0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        rounded_entropy = round(entropy, 2)
        return rounded_entropy
    
    def _kurtois(self):
        data = self.train_dataset.data_input.numpy().flatten()
        mean = np.mean(data)
        std_dev = np.std(data)
        kurtosis = np.sum((data - mean) ** 4) / (len(data) * std_dev ** 4)
        rounded_kurtosis = round(kurtosis, 2)
        return rounded_kurtosis
    
    def _class_distribution(self):
        labels = self.train_dataset.data_label.tolist()
        counts = {}
        for elem in labels:
            counts[elem] = counts.get(elem, 0) + 1
        return counts
    
    def _get_data_range(self):
        data = self.train_dataset.data_input
        
        # Check if the tensor is empty
        if data.numel() == 0:
            print("Warning: The data tensor is empty.")
            return {"min": None, "max": None}
        
        # Check for NaN values
        if torch.isnan(data).any():
            print("Warning: The data contains NaN values. Removing NaN for min/max calculation.")
            data = data[~torch.isnan(data)]
        
        # Check if there's any data left after removing NaN
        if data.numel() == 0:
            print("Warning: All data values are NaN.")
            return {"min": None, "max": None}
        
        # Ensure the data is a float type
        data = data.float()
        
        data_min = torch.min(data).item()
        data_max = torch.max(data).item()
        
        # Final check for NaN
        if np.isnan(data_min) or np.isnan(data_max):
            print("Warning: Min or Max is NaN after calculation.")
            return {"min": None, "max": None}
        
        return {"min": data_min, "max": data_max}
    
    def _brisque(self):
        data = self.train_dataset.data_input
        brisque = piq.brisque(data)
        rounded_brisque = round(brisque.item(), 2)
        return rounded_brisque
    
    def _total_variation(self):
        data = self.train_dataset.data_input
        total_variation = piq.total_variation(data)
        rounded_total_variation = round(total_variation.item(), 2)
        return rounded_total_variation
    
    def _image_sharpness(self, image):
        
        """
        Calculate sharpness of an image using the variance of the Laplacian.
        Higher values indicate sharper images.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.ndim == 3:
            image = np.mean(image, axis=0)  # Convert to grayscale if it's a color image
        
        laplacian = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        
        conv = np.abs(np.convolve(image.flatten(), laplacian.flatten(), mode='same'))
        var = np.var(conv)

        return var
    
    def _dataset_sharpness(self):
        dataset = self.train_dataset
        sharpness_scores = [self._image_sharpness(img) for img in dataset.data_input]
        avg_sharpness = np.mean(sharpness_scores)
        rounded_sharpness = round(avg_sharpness, 2)
        return rounded_sharpness
    
    def _plot_class_distribution(self):
        
        plt.figure()

        classes = self.train_dataset.data_label.unique().tolist()
        counts = []
        for c in classes:
            counts.append(self.train_dataset.data_label.tolist().count(c))
        
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
    
    def _plot_data_sample(self):
        plt.figure(figsize=(15, 3))  

        data = self.train_dataset.data_input
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
    
    def _plot_data_distribution(self):
        # Flatten the pixel values
        data = random.sample(self.train_dataset.data_input.tolist(), 10)
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
    
    def _plot_class_variance(self):
        # Get the unique classes in the dataset
        classes = torch.unique(self.train_dataset.data_label).tolist()

        # Create a dictionary to store the variance for each class
        class_variance = {}

        # Calculate the variance for each class
        for c in classes:
            # Create a boolean mask for the current class
            class_mask = self.train_dataset.data_label == c

            # Apply the mask to get the data for the current class
            class_data = self.train_dataset.data_input[class_mask]

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

     


    
    
    
    

        
    
        
        
    

    
    
        