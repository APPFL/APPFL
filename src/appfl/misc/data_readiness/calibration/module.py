"""Evaluate stored predictions through CADRE."""

from typing import Any, Dict, Optional

from ..base_cadremodule import BaseCADREModule
from .metrics import build_calibration_report


class SubgroupCalibrationCADREModule(BaseCADREModule):
    """Read predictions, binary outcomes, and subgroup labels from a dataset."""

    def metric(
        self,
        n_bins: int = 10,
        min_cell_count: int = 5,
        release_n_bins: Optional[int] = None,
        min_outcome_count: int = 0,
    ) -> Dict[str, Any]:
        dataset = self.train_dataset
        if any(
            not hasattr(dataset, key) for key in ("predictions", "outcomes", "groups")
        ):
            raise ValueError("dataset needs predictions, outcomes, and groups")
        return {
            "subgroup_calibration": build_calibration_report(
                dataset.predictions,
                dataset.outcomes,
                dataset.groups,
                n_bins=n_bins,
                min_cell_count=min_cell_count,
                release_n_bins=release_n_bins,
                min_outcome_count=min_outcome_count,
            )
        }

    def rule(self, metric_result: Dict[str, Any], **kwargs) -> bool:
        return False

    def remedy(
        self, metric_result: Dict[str, Any], logger=None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluation does not alter the dataset."""
        return {"ai_ready_dataset": self.train_dataset, "metadata": None}
