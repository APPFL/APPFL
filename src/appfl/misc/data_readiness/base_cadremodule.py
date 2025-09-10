import abc
from typing import Dict, Any


class BaseCADREModule(abc.ABC):
    def __init__(self, train_dataset: Any, **kwargs):
        """
        Base class for dataset analysis with customizable metric and rule methods.

        Args:
        - train_dataset: Dataset to be analyzed.
        - kwargs: Additional parameters for subclasses.
        """
        self.train_dataset = train_dataset
        self.kwargs = kwargs  # Store kwargs for subclasses if needed

    @abc.abstractmethod
    def metric(self) -> Dict[str, Any]:
        """Compute and return metric results."""
        pass

    @abc.abstractmethod
    def rule(self, metric_result: Dict[str, Any], **kwargs) -> bool:
        """
        Check if the rule condition is met.

        Args:
        - metric_result: Dictionary containing computed metric values.
        - kwargs: Additional parameters for subclass implementations.

        Returns:
        - True if the rule condition is met, False otherwise.
        """
        pass

    @abc.abstractmethod
    def remedy(self, metric_result: Dict[str, Any], **kwargs) -> Any:
        """
        Apply a remedy based on metric results.

        Args:
        - metric_result: Dictionary containing computed metric values.
        - kwargs: Additional parameters for subclass implementations.

        Returns:
        - Modified dataset.
        """
        pass
