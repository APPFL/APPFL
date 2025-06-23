import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from appfl.misc.data_readiness import BaseCADREModule

class CADREModuleLOFOutlier(BaseCADREModule):
    def __init__(self, train_dataset, contamination=0.05, n_neighbors=20, **kwargs):
        """
        Initialize the LOF-based outlier detector.

        Args:
        - train_dataset: The dataset to analyze.
        - contamination: Expected proportion of outliers (default: 5%).
        - n_neighbors: Number of neighbors for LOF (default: 20).
        - kwargs: Additional keyword arguments.
        """
        super().__init__(train_dataset, **kwargs)
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def metric(self, **kwargs):
        """
        Compute LOF outlier scores on flattened images.

        Returns:
        - A dictionary containing:
            - "outlier_indices": Indices of detected outliers.
            - "outlier_scores": LOF anomaly scores (lower = more anomalous).
            - "mean_outlier_score": Mean anomaly score across all samples.
        """
        # Flatten all images into vectors
        flattened_images = []
        for i in range(len(self.train_dataset)):
            img, _ = self.train_dataset[i]
            img_np = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            flattened_images.append(img_np.flatten())
        X = np.vstack(flattened_images)

        # Detect outliers using LOF
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False
        )
        outlier_labels = lof.fit_predict(X)
        outlier_indices = np.where(outlier_labels == -1)[0].tolist()
        outlier_scores = -lof.negative_outlier_factor_  # Convert to positive scores

        return {
            "mean_outlier_score": float(np.mean(outlier_scores)),
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores.tolist(),
            
        }

    def rule(self, metric_result, **kwargs):
        """
        Return indices of detected outliers.
        """
        return metric_result["outlier_indices"]

    def remedy(self, metric_result, logger, **kwargs):
        """
        Remove outliers from the dataset.
        """
        all_indices = set(range(len(self.train_dataset)))
        outlier_indices = set(metric_result["outlier_indices"])
        keep_indices = list(all_indices - outlier_indices)

        logger.info(f"Removed {len(outlier_indices)} LOF-based outliers (contamination={self.contamination}).")

        cleaned_dataset = [self.train_dataset[i] for i in keep_indices]

        return {
            "ai_ready_dataset": cleaned_dataset,
            "metadata": {
                "removed_indices": list(outlier_indices),
                "removed_scores": [metric_result["outlier_scores"][i] for i in outlier_indices],
                "mean_outlier_score": metric_result["mean_outlier_score"]
            }
        }