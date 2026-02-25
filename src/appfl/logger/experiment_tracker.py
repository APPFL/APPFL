from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from appfl.logger.utils import _remap_server_wandb_payload


@dataclass
class TrackerConfig:
    backend: str
    project_name: str
    run_name: str
    log_dir: str
    experiment_seed: str
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_tags: list[str] | None = None


def _cfg_get(payload: dict, path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur: Any = payload
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        return default
    return default if cur is None else cur


def _extract_tracker_config(config: DictConfig | dict) -> TrackerConfig:
    if isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    else:
        cfg = dict(config)

    backend = str(_cfg_get(cfg, "logging.backend", "file")).lower()
    experiment_name = str(_cfg_get(cfg, "experiment.name", "appfl-sim"))
    run_name = str(_cfg_get(cfg, "logging.name", experiment_name))
    project_name = experiment_name
    log_dir = str(_cfg_get(cfg, "logging.path", "./logs"))
    experiment_seed = str(_cfg_get(cfg, "experiment.seed", 0))
    wandb_entity = str(_cfg_get(cfg, "logging.configs.wandb_entity", ""))
    wandb_mode = str(_cfg_get(cfg, "logging.configs.wandb_mode", "online")).lower()
    wandb_tags_cfg = _cfg_get(cfg, "logging.configs.wandb_tags", None)
    wandb_tags: list[str] | None
    if isinstance(wandb_tags_cfg, list):
        wandb_tags = [str(tag).strip() for tag in wandb_tags_cfg if str(tag).strip()]
    elif isinstance(wandb_tags_cfg, str):
        wandb_tags = [
            token.strip() for token in wandb_tags_cfg.split(",") if token.strip()
        ]
    else:
        wandb_tags = None
    return TrackerConfig(
        backend=backend,
        project_name=project_name,
        run_name=run_name,
        log_dir=log_dir,
        experiment_seed=experiment_seed,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        wandb_tags=wandb_tags,
    )


def _has_wandb_api_credential() -> bool:
    if os.getenv("WANDB_API_KEY"):
        return True

    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth is not None and bool(auth[2])
    except Exception:
        return False


class ExperimentTracker:
    """Track experiment metrics for file-only, TensorBoard, or Weights & Biases backends."""

    def __init__(self, config: DictConfig | dict, run_id: str):
        cfg = _extract_tracker_config(config)
        run_id_text = str(run_id).strip()
        if run_id_text == "":
            raise ValueError("run_id must be a non-empty string.")
        self.backend = cfg.backend
        self._writer = None
        self._wandb = None
        self._run = None
        self._metrics_json_path = None
        self._metrics_records: list[dict[str, Any]] = []

        if self.backend in {"none", "file", "console"}:
            run_dir = Path(cfg.log_dir) / cfg.project_name / cfg.run_name / run_id_text
            run_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_json_path = run_dir / "metrics.json"
            if self._metrics_json_path.exists():
                try:
                    content = json.loads(
                        self._metrics_json_path.read_text(encoding="utf-8")
                    )
                    if isinstance(content, list):
                        self._metrics_records = content
                except Exception:
                    self._metrics_records = []
            return

        if self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "TensorBoard logging requested but tensorboard is not installed. "
                    "Install with: pip install tensorboard"
                ) from e

            run_dir = Path(cfg.log_dir) / cfg.project_name / cfg.run_name / run_id_text
            run_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(run_dir))
            return

        if self.backend == "wandb":
            try:
                import wandb
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "WandB logging requested but wandb is not installed. "
                    "Install with: pip install wandb"
                ) from e

            if cfg.wandb_mode != "offline" and not _has_wandb_api_credential():
                raise RuntimeError(
                    "WandB logging requires CLI authentication. "
                    "Run `wandb login` (or set WANDB_API_KEY) before starting simulation."
                )

            init_kwargs = {
                "project": cfg.project_name,
                "name": cfg.run_name,
                "dir": cfg.log_dir,
                "mode": cfg.wandb_mode,
                "reinit": True,
            }
            tags = list(cfg.wandb_tags or [])
            tags.append(f"seed:{cfg.experiment_seed}")
            # Keep order stable while avoiding duplicate tags.
            init_kwargs["tags"] = list(dict.fromkeys(tags))
            if cfg.wandb_entity:
                init_kwargs["entity"] = cfg.wandb_entity

            self._wandb = wandb
            self._run = wandb.init(**init_kwargs)
            return

        raise ValueError(
            "logging.backend must be one of: none, file, console, tensorboard, wandb"
        )

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(k): ExperimentTracker._to_json_compatible(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ExperimentTracker._to_json_compatible(v) for v in value]
        if isinstance(value, (str, bool)) or value is None:
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _flatten_numeric_metrics(
        metrics: dict[str, Any], prefix: str = ""
    ) -> dict[str, float]:
        flat: dict[str, float] = {}
        for key, val in metrics.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(val, dict):
                flat.update(
                    ExperimentTracker._flatten_numeric_metrics(val, prefix=name)
                )
                continue
            if isinstance(val, (int, float)):
                flat[name] = float(val)
                continue
            if hasattr(val, "item"):
                try:
                    scalar = val.item()
                    if isinstance(scalar, (int, float)):
                        flat[name] = float(scalar)
                except Exception:
                    continue
        return flat

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if not metrics:
            return
        if self.backend in {"none", "file", "console"}:
            if self._metrics_json_path is None:
                return
            payload = {
                "round": int(step),
                "metrics": self._to_json_compatible(metrics),
            }
            self._metrics_records.append(payload)
            self._metrics_json_path.write_text(
                json.dumps(self._metrics_records, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            return
        if self.backend == "tensorboard" and self._writer is not None:
            for key, val in self._flatten_numeric_metrics(metrics).items():
                self._writer.add_scalar(key, float(val), global_step=int(step))
            self._writer.flush()
            return
        if self.backend == "wandb" and self._wandb is not None:
            payload = self._flatten_numeric_metrics(metrics)
            payload["round"] = int(step)
            self._wandb.log(_remap_server_wandb_payload(payload), step=int(step))

    def close(self) -> None:
        if self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.backend == "wandb" and self._run is not None:
            self._run.finish()
            self._run = None


def create_experiment_tracker(
    config: DictConfig | dict, run_id: str
) -> ExperimentTracker:
    return ExperimentTracker(config, run_id=run_id)
