from base_dragent import BaseDRAgent
import random
import numpy as np

class DRAgent(BaseDRAgent):
    def __init__(self, train_dataset, **kwargs):
        """
        DRAgent specializing in noise detection with configurable parameters.
        """
        super().__init__(train_dataset, **kwargs)
    
    def metric(self):
        # Determine how to retrieve data input and labels based on dataset attributes
        if hasattr(self.train_dataset, 'data_label'):
            data_labels = self.train_dataset.data_label.tolist()
        else:
            try:

                data_labels = [label.item() for _, label in self.train_dataset]
            except:

                data_labels = [label for _, label in self.train_dataset]
        
        # Count occurrences of each class
        counts = {}
        for elem in data_labels:
            counts[elem] = counts.get(elem, 0) + 1
        
        # Check if only one class exists
        num_classes = len(counts)
        if num_classes == 1:
            return float('inf')  # Or return a specific value like 0, or None, or raise an error
        
        # Calculate imbalance degree
        total_elements = len(data_labels)
        actual_proportions = np.array([counts[elem] / total_elements for elem in sorted(counts.keys())])
        balanced_proportions = np.array([1 / num_classes] * num_classes)
        euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)
        
        # Normalize for 2-class problems
        if num_classes == 2:
            # Calculate the maximum possible imbalance
            max_imbalance = np.linalg.norm(np.array([1/total_elements, (total_elements-1)/total_elements]) - np.array([0.5, 0.5]))
            
            # Normalize the euclidean distance
            normalized_distance = euclidean_distance / max_imbalance
            
            return {"class_imbalance":round(normalized_distance,2)}
        
        return {"class_imbalance":round(euclidean_distance,2)}
    
    def rule(self, metric_result: float, threshold = 0.3):
        """
        Check if the rule condition is met.
        
        Args:
        - metric_result: A dictionary containing computed metric results.
        
        Returns:
        - True if the rule condition is met, False otherwise.
        """
        return metric_result['class_imbalance'] > threshold
    
    def remedy(self, metric_result, logger, proportion=0.0001):
        """
        Apply the remedy to the dataset.
        
        Args:
        - logger: Logger instance for logging modifications.
        
        Returns:
        - Modified dataset.
        """
        if self.rule(metric_result):
            num_samples = int(proportion * len(self.train_dataset))
            self.train_dataset = random.sample(self.train_dataset, num_samples)
            logger.info("Data modified based on user-defined rule")
        return self.train_dataset
