from base_dragent import BaseDRAgent
import torch
import random
from typing import Dict, Any

class DRAgent(BaseDRAgent):
    def __init__(self, train_dataset, **kwargs):
        """
        DRAgent specializing in noise detection with configurable parameters.
        """
        super().__init__(train_dataset, **kwargs)
    
    def metric(self):
        if hasattr(self.train_dataset, 'data_input'):
            data_input = self.train_dataset.data_input
        else:
            data_input = torch.stack([input_data for input_data, _ in self.train_dataset])
        
        magnitudes = [torch.mean(sample[0] ** 2).item() for sample in data_input]
        mean = sum(magnitudes) / len(magnitudes)
        return {"mean": round(mean, 2)}
    
    def rule(self, metric_result: float, threshold = 0.3):
        """
        Check if the rule condition is met.
        
        Args:
        - metric_result: A dictionary containing computed metric results.
        
        Returns:
        - True if the rule condition is met, False otherwise.
        """
        return metric_result['mean'] > threshold
    
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
