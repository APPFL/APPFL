import abc
from typing import Any


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
    def metric(self) -> dict[str, Any]:
        """Compute and return metric results."""

    @abc.abstractmethod
    def rule(self, metric_result: dict[str, Any], **kwargs) -> bool:
        """
        Check if the rule condition is met.

        Args:
        - metric_result: Dictionary containing computed metric values.
        - kwargs: Additional parameters for subclass implementations.

        Returns:
        - True if the rule condition is met, False otherwise.
        """

    @abc.abstractmethod
    def remedy(self, metric_result: dict[str, Any], **kwargs) -> Any:
        """
        Apply a remedy based on metric results.

        Args:
        - metric_result: Dictionary containing computed metric values.
        - kwargs: Additional parameters for subclass implementations.

        Returns:
        - Modified dataset.
        """
