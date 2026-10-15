"""
CADRE module for genomic sequence data quality assessment using DRISEE scores.

Subclasses BaseCADREModule following the APPFL PR #440 pattern.
Unlike standard CADRE modules, this one does not need (input, label) pairs —
it reads DRISEE scores directly from the site's AIDRIN CSV.

Usage in server data_readiness_configs:
    cadremodule_configs:
      cadremodule_path: /path/to/cadre_module_drisee.py
      cadremodule_name: CADREModuleDRISEE
      cadremodule_kwargs:
        aidrin_csv:       /path/to/site_aidrin_inputs.csv
        fasta_dir:        /path/to/fasta
        drisee_threshold: 20.0
        max_files:        1
      remedy_action: false   # exclusion happens inside trainer, not here
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleDRISEE(BaseCADREModule):
    """
    Data-readiness CADRE module for genomic FASTA datasets assessed by DRISEE.

    metric()  — returns per-file DRISEE scores and site-level summary
    rule()    — True when any file exceeds drisee_threshold (site not fully clean)
    remedy()  — no-op: contaminated files are excluded by the trainer at init time
    """

    def __init__(self, train_dataset, **kwargs):
        super().__init__(train_dataset, **kwargs)

    def metric(
        self,
        aidrin_csv: str = "",
        fasta_dir: str = "",
        drisee_threshold: float = 20.0,
        max_files: Optional[int] = None,
        current_exclude: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            # Normal package import (this module loaded as
            # appfl.algorithm.trainer.cadre_module_drisee).
            from .aidrin_checker import check_site_data
        except ImportError:
            # Loaded dynamically by file path (server `cadremodule_path`);
            # `create_instance_from_file` puts this file's own directory on
            # sys.path, so the flat import resolves there instead.
            from aidrin_checker import check_site_data

        if not aidrin_csv or not fasta_dir:
            return {"error": "aidrin_csv and fasta_dir must be provided"}

        _, _, report = check_site_data(
            site=Path(fasta_dir).parent.name,
            aidrin_csv=Path(aidrin_csv),
            fasta_dir=Path(fasta_dir),
            current_exclude=current_exclude or [],
            max_files=max_files,
            drisee_threshold=drisee_threshold,
        )
        return {
            "n_clean":        report.get("n_clean", 0),
            "n_contaminated": report.get("n_contaminated", 0),
            "site_ready":     report.get("site_ready", True),
            "drisee_threshold": drisee_threshold,
        }

    def rule(self, metric_result: Dict[str, Any], **kwargs) -> bool:
        """True when any contamination was found."""
        return metric_result.get("n_contaminated", 0) > 0

    def remedy(self, metric_result: Dict[str, Any], logger, **kwargs) -> Dict[str, Any]:
        """No active remedy — contaminated files are excluded by the trainer."""
        if self.rule(metric_result) and logger:
            logger.info(
                "[CADREModuleDRISEE] Contaminated files detected; "
                "exclusion is handled by Evo2PEFTTrainer at init time."
            )
        return {"ai_ready_dataset": self.train_dataset, "metadata": metric_result}
