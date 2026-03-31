"""
Run Manager - Organizes experiment runs with consistent folder structure.

Each run creates a timestamped folder containing:
- config.yaml (copy of configuration used)
- checkpoints/ (model checkpoints)
- logs/ (tensorboard/wandb logs)
- evaluation/ (created by evaluate_model.py)
- run_info.yaml (metadata about the run)

Example structure:
    runs/
    ├── pretrain_2024-02-05_14-30-25_my_experiment/
    │   ├── config.yaml
    │   ├── run_info.yaml
    │   ├── checkpoints/
    │   │   ├── model_latest.pt      # Raw weights, updated each epoch
    │   │   ├── last.ckpt            # Lightning checkpoint
    │   │   └── final_model.pt       # Raw weights at end of training
    │   ├── logs/
    │   └── evaluation/
    └── finetune_2024-02-06_09-00-00_classification/
        └── ...
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import yaml


class RunManager:
    def __init__(
        self,
        run_type: str,
        experiment_name: Optional[str] = None,
        base_dir: str = "runs",
        config_path: Optional[str] = None,
    ):
        self.run_type = run_type
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config_path = config_path

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = self._create_run_name()
        self.run_dir = self.base_dir / self.run_name

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.evaluation_dir = self.run_dir / "evaluation"

        self._create_directories()

        if config_path:
            self._copy_config(config_path)

    def _create_run_name(self) -> str:
        parts = [self.run_type, self.timestamp]
        if self.experiment_name:
            safe_name = self.experiment_name.replace(" ", "_").replace("/", "-")
            parts.append(safe_name)
        return "_".join(parts)

    def _create_directories(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run Directory: {self.run_dir}")
        print(f"{'='*60}")

    def _copy_config(self, config_path: str):
        src = Path(config_path)
        if src.exists():
            dst = self.run_dir / "config.yaml"
            shutil.copy2(src, dst)
            print(f"Config saved to: {dst}")

    def save_config(self, config: Dict[str, Any]):
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to: {config_path}")

    def save_run_info(self, info: Dict[str, Any]):
        run_info = {
            "run_type": self.run_type,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "experiment_name": self.experiment_name,
            "run_dir": str(self.run_dir),
            **info
        }

        info_path = self.run_dir / "run_info.yaml"
        with open(info_path, 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
        print(f"Run info saved to: {info_path}")

    def get_evaluation_dir(self, eval_name: Optional[str] = None) -> Path:
        if eval_name:
            eval_dir = self.evaluation_dir / eval_name
        else:
            eval_dir = self.evaluation_dir

        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    @classmethod
    def from_existing_run(cls, run_dir: str) -> "RunManager":
        run_path = Path(run_dir)
        if not run_path.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        run_name = run_path.name
        parts = run_name.split("_")
        run_type = parts[0] if parts else "unknown"

        instance = cls.__new__(cls)
        instance.run_type = run_type
        instance.run_name = run_name
        instance.run_dir = run_path
        instance.base_dir = run_path.parent
        instance.checkpoint_dir = run_path / "checkpoints"
        instance.log_dir = run_path / "logs"
        instance.evaluation_dir = run_path / "evaluation"
        instance.timestamp = "_".join(parts[1:3]) if len(parts) >= 3 else ""
        instance.experiment_name = "_".join(parts[3:]) if len(parts) > 3 else None
        instance.config_path = None

        return instance

    def get_checkpoint_for_mode(self, mode: str) -> Optional[Path]:
        """
        Get the best checkpoint for a given evaluation mode.

        For pretrain: prefer raw weights (final_model.pt > model_latest.pt),
            fall back to last.ckpt (Lightning format, handled by evaluate_model).
        For finetune: prefer last.ckpt (contains encoder + heads),
            fall back to encoder_latest.pt / final_encoder.pt (encoder only).
        """
        if mode == "finetune":
            # Finetune needs the full Lightning checkpoint (encoder + heads)
            last = self.checkpoint_dir / "last.ckpt"
            if last.exists():
                return last
            # Fallback: raw encoder (evaluate_finetune can't use this directly,
            # but it's better than nothing)
            for name in ["final_encoder.pt", "encoder_latest.pt"]:
                p = self.checkpoint_dir / name
                if p.exists():
                    return p
        else:
            # Pretrain: prefer raw weights (simpler, faster to load)
            for name in ["final_model.pt", "model_latest.pt"]:
                p = self.checkpoint_dir / name
                if p.exists():
                    return p
            # Fallback: Lightning checkpoint
            last = self.checkpoint_dir / "last.ckpt"
            if last.exists():
                return last

        # Last resort: any .ckpt or .pt
        pts = list(self.checkpoint_dir.glob("*.pt"))
        if pts:
            return max(pts, key=lambda p: p.stat().st_mtime)
        ckpts = list(self.checkpoint_dir.glob("*.ckpt"))
        if ckpts:
            return max(ckpts, key=lambda p: p.stat().st_mtime)

        return None

    def __str__(self) -> str:
        return f"RunManager(run_dir={self.run_dir})"

    def __repr__(self) -> str:
        return self.__str__()


def find_latest_run(run_type: str = None, base_dir: str = "runs") -> Optional[Path]:
    base = Path(base_dir)
    if not base.exists():
        return None

    runs = [r for r in base.iterdir() if r.is_dir()]
    if run_type:
        runs = [r for r in runs if r.name.startswith(run_type)]

    if not runs:
        return None

    return max(runs, key=lambda p: p.stat().st_mtime)


def list_runs(run_type: str = None, base_dir: str = "runs") -> list:
    base = Path(base_dir)
    if not base.exists():
        return []

    runs = [r for r in base.iterdir() if r.is_dir()]
    if run_type:
        runs = [r for r in runs if r.name.startswith(run_type)]

    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
