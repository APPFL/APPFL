"""Subgroup calibration from client-side totals."""

from .metrics import aggregate_calibration_reports, build_calibration_report
from .module import SubgroupCalibrationCADREModule

__all__ = [
    "SubgroupCalibrationCADREModule",
    "aggregate_calibration_reports",
    "build_calibration_report",
]
