#!/usr/bin/env python
"""
List and manage experiment runs.

Usage:
    python list_runs.py                    # List all runs
    python list_runs.py --type pretrain    # List only pretrain runs
    python list_runs.py --details RUN_DIR  # Show details for a specific run
"""

import argparse
from pathlib import Path
import yaml

from utils.run_manager import list_runs, find_latest_run


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_dir_size(path: Path) -> int:
    """Get total size of directory."""
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total


def load_run_info(run_dir: Path) -> dict:
    """Load run_info.yaml if exists."""
    info_path = run_dir / "run_info.yaml"
    if info_path.exists():
        with open(info_path) as f:
            return yaml.safe_load(f)
    return {}


def show_run_details(run_dir: str):
    """Show detailed info for a specific run."""
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Run directory not found: {run_dir}")
        return
    
    info = load_run_info(run_path)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_path.name}")
    print(f"{'='*60}")
    print(f"Path: {run_path}")
    print(f"Size: {format_size(get_dir_size(run_path))}")
    
    if info:
        print(f"\nRun Info:")
        print(f"  Status: {info.get('status', 'unknown')}")
        print(f"  Type: {info.get('run_type', 'unknown')}")
        
        if 'model_params' in info:
            print(f"  Model params: {info['model_params']:,}")
        if 'train_samples' in info:
            print(f"  Train samples: {info['train_samples']:,}")
        if 'val_samples' in info:
            print(f"  Val samples: {info['val_samples']:,}")
        if 'best_val_loss' in info and info['best_val_loss']:
            print(f"  Best val loss: {info['best_val_loss']:.4f}")
        if 'best_metric' in info and info['best_metric']:
            print(f"  Best metric: {info['best_metric']:.4f}")
        if 'task' in info:
            print(f"  Task: {info['task']}")
        if 'pretrain_run' in info and info['pretrain_run']:
            print(f"  Pretrain run: {info['pretrain_run']}")
        
        print(f"\nMasking config:")
        print(f"  Contiguous: {info.get('contiguous_masking', 'N/A')}")
        print(f"  Block sizes: {info.get('block_sizes', 'N/A')}")
        print(f"  Peak-biased: {info.get('peak_bias_enabled', 'N/A')}")
        print(f"  Mask ratio: {info.get('mask_ratio', 'N/A')}")
    
    # List contents
    print(f"\nContents:")
    
    checkpoints = list((run_path / "checkpoints").glob("*")) if (run_path / "checkpoints").exists() else []
    if checkpoints:
        print(f"  Checkpoints ({len(checkpoints)} files):")
        for ckpt in sorted(checkpoints)[:5]:
            print(f"    - {ckpt.name}")
        if len(checkpoints) > 5:
            print(f"    ... and {len(checkpoints) - 5} more")
    
    if (run_path / "evaluation").exists():
        evals = list((run_path / "evaluation").iterdir())
        print(f"  Evaluations ({len(evals)}):")
        for e in sorted(evals)[:5]:
            print(f"    - {e.name}")
    
    config_path = run_path / "config.yaml"
    if config_path.exists():
        print(f"  Config: config.yaml")


def main():
    parser = argparse.ArgumentParser(description="List and manage experiment runs")
    parser.add_argument('--runs_dir', type=str, default='runs', help='Base runs directory')
    parser.add_argument('--type', type=str, choices=['pretrain', 'finetune'], help='Filter by run type')
    parser.add_argument('--details', type=str, help='Show details for a specific run')
    parser.add_argument('--latest', action='store_true', help='Show only the latest run')
    
    args = parser.parse_args()
    
    if args.details:
        show_run_details(args.details)
        return
    
    if args.latest:
        latest = find_latest_run(run_type=args.type, base_dir=args.runs_dir)
        if latest:
            show_run_details(str(latest))
        else:
            print("No runs found.")
        return
    
    # List all runs
    runs = list_runs(run_type=args.type, base_dir=args.runs_dir)
    
    if not runs:
        print(f"No runs found in {args.runs_dir}/")
        return
    
    print(f"\n{'='*80}")
    print(f"{'Run Name':<55} {'Status':<10} {'Size':<10}")
    print(f"{'='*80}")
    
    for run_dir in runs:
        info = load_run_info(run_dir)
        status = info.get('status', '?')
        size = format_size(get_dir_size(run_dir))
        
        # Truncate long names
        name = run_dir.name
        if len(name) > 53:
            name = name[:50] + "..."
        
        # Color-code status
        status_str = status[:8]
        
        print(f"{name:<55} {status_str:<10} {size:<10}")
    
    print(f"\nTotal: {len(runs)} runs")
    print(f"\nTip: Use --details <run_dir> to see more info")
    print(f"     Use --latest to see the most recent run")


if __name__ == "__main__":
    main()
