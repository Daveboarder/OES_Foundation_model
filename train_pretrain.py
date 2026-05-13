"""
Pre-training script for LIBS Foundation Model.

Creates a timestamped run directory, generates synthetic data,
trains the transformer with MIP, and saves checkpoints.

Usage:
    uv run python train_pretrain.py --config config/config_a100.yaml --experiment_name big_pretrain
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_generator import SyntheticLIBSGenerator
from data.libs_pipeline import build_dataset_from_config
from models.libs_transformer import LIBSTransformer
from training.pretrain import LIBSPretrainModule, PretrainDataModule
from utils.run_manager import RunManager


class SaveRawModelCallback(Callback):
    """Save raw model weights every N epochs for easy mid-training evaluation."""
    def __init__(self, save_path: str, save_every_n_epochs: int = 1):
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(pl_module.model.state_dict(), self.save_path)
            info_path = str(self.save_path).replace('.pt', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"epoch: {trainer.current_epoch + 1}\n")
                f.write(f"global_step: {trainer.global_step}\n")
                val_loss = trainer.callback_metrics.get('val/loss')
                if val_loss is not None:
                    f.write(f"val_loss: {val_loss.item():.6f}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _generate_legacy_synthetic(config: dict, seed: int):
    """Original 5-class Gaussian-peak synthetic data (kept for sanity comparison)."""
    print("Generating legacy synthetic data...")
    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )
    train_spectra, _, _ = generator.generate_dataset(
        n_samples=config['data']['synthetic']['train_samples'], return_labels=True,
    )
    generator_val = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed + 1000,
    )
    val_spectra, _, _ = generator_val.generate_dataset(
        n_samples=config['data']['synthetic']['val_samples'], return_labels=True,
    )
    return train_spectra, val_spectra


def _generate_libs_pipeline(config: dict, libs_config_path: str, seed: int):
    """Realistic physics-based synthesis (Daveboarder/Element_Identification heritage).

    The full dataset is materialised once (cached to HDF5), then split into
    train/val by `data.synthetic.val_fraction` (default 0.1). n_bins is
    overridden in-place to match the actual wavelength array length."""
    print(f"Generating LIBS-pipeline data from {libs_config_path}...")
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})['seed'] = seed

    ds = build_dataset_from_config(libs_cfg)
    spectra = ds.spectra.astype(np.float32)
    if spectra.size == 0:
        raise RuntimeError("LIBS pipeline produced no spectra — check sample types / DB.")

    # Override model n_bins to match the real wavelength axis (e.g. 69712)
    actual_n_bins = spectra.shape[1]
    if config['data']['n_bins'] != actual_n_bins:
        print(f"Overriding n_bins: {config['data']['n_bins']} -> {actual_n_bins}")
        config['data']['n_bins'] = actual_n_bins
        config['model']['max_seq_len'] = actual_n_bins + 1

    val_frac = config['data']['synthetic'].get('val_fraction', 0.1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(spectra))
    n_val = max(1, int(len(spectra) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return spectra[train_idx], spectra[val_idx]


def generate_data(config: dict, seed: int = 42, libs_config_path: str | None = None):
    if libs_config_path:
        return _generate_libs_pipeline(config, libs_config_path, seed)
    train_spectra, val_spectra = _generate_legacy_synthetic(config, seed)
    print(f"Generated {len(train_spectra)} training samples")
    print(f"Generated {len(val_spectra)} validation samples")
    return train_spectra, val_spectra


def create_model(config: dict) -> LIBSTransformer:
    model = LIBSTransformer(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
    )
    print(f"Created model with {model.num_parameters:,} parameters")
    return model


def main(args):
    config = load_config(args.config)

    run_mgr = RunManager(
        run_type="pretrain",
        experiment_name=args.experiment_name,
        base_dir=args.runs_dir,
        config_path=args.config,
    )

    pl.seed_everything(args.seed)

    train_spectra, val_spectra = generate_data(
        config, seed=args.seed, libs_config_path=args.libs_data_config,
    )
    print(f"Train: {len(train_spectra)} samples | Val: {len(val_spectra)} samples | n_bins: {train_spectra.shape[1]}")

    if args.save_data:
        data_dir = run_mgr.run_dir / "data"
        data_dir.mkdir(exist_ok=True)
        np.save(data_dir / 'train_spectra.npy', train_spectra)
        np.save(data_dir / 'val_spectra.npy', val_spectra)
        print(f"Saved data to {data_dir}")

    model = create_model(config)

    pretrain_module = LIBSPretrainModule(
        model=model,
        learning_rate=config['pretrain']['learning_rate'],
        weight_decay=config['pretrain']['weight_decay'],
        warmup_epochs=config['pretrain']['warmup_epochs'],
        max_epochs=config['pretrain']['epochs'],
        min_lr=config['pretrain']['min_lr'],
    )

    # Masking config
    contiguous_masking = config['pretrain'].get('contiguous_masking', False) or args.contiguous_masking
    block_sizes_config = config['pretrain'].get('block_sizes', None)
    block_sizes = block_sizes_config if block_sizes_config else [config['pretrain'].get('contiguous_mask_size', 50)]
    peak_bias_enabled = config['pretrain'].get('peak_bias_enabled', False)
    peak_bias_ratio = config['pretrain'].get('peak_bias_ratio', 0.5)
    peak_threshold = config['pretrain'].get('peak_threshold', 0.2)

    print(f"\nMasking Configuration:")
    print(f"  Contiguous: {contiguous_masking}")
    print(f"  Block sizes: {block_sizes}")
    print(f"  Mask ratio: {config['pretrain']['mask_ratio']}")
    print(f"  Peak-biased: {peak_bias_enabled} (ratio: {peak_bias_ratio}, threshold: {peak_threshold})")

    data_module = PretrainDataModule(
        train_spectra=train_spectra,
        val_spectra=val_spectra,
        batch_size=config['pretrain']['batch_size'],
        mask_ratio=config['pretrain']['mask_ratio'],
        contiguous_masking=contiguous_masking,
        block_sizes=block_sizes,
        peak_bias_enabled=peak_bias_enabled,
        peak_bias_ratio=peak_bias_ratio,
        peak_threshold=peak_threshold,
        num_workers=args.num_workers,
    )

    # Logger
    logger_type = config['logging'].get('logger', 'tensorboard')

    if logger_type == 'wandb':
        wandb_config = config['logging'].get('wandb', {})
        logger = WandbLogger(
            project=wandb_config.get('project', 'libs-foundation-model'),
            entity=wandb_config.get('entity'),
            name=run_mgr.run_name,
            tags=wandb_config.get('tags', []) + ['pretrain'],
            save_dir=str(run_mgr.log_dir),
            config={
                'model': config['model'],
                'pretrain': config['pretrain'],
                'data': config['data'],
                'run_dir': str(run_mgr.run_dir),
            },
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(run_mgr.log_dir),
            name='',
            version='',
        )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_mgr.checkpoint_dir),
        filename='best',
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        save_last=True,
        auto_insert_metric_name=False,
    )

    raw_model_callback = SaveRawModelCallback(
        save_path=str(run_mgr.checkpoint_dir / 'model_latest.pt'),
        save_every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor, raw_model_callback]
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val/loss', patience=10, mode='min', verbose=True,
        ))

    # Trainer
    trainer_kwargs = dict(
        accelerator=config['device']['accelerator'],
        devices=1,
        precision=config['device']['precision'],
        max_epochs=config['pretrain']['epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        val_check_interval=config['logging']['val_check_interval'],
        gradient_clip_val=1.0,
        deterministic=True,
    )
    # Gradient accumulation (for large effective batch on limited VRAM)
    accumulate = config['pretrain'].get('accumulate_grad_batches', None)
    if accumulate and accumulate > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate

    trainer = pl.Trainer(**trainer_kwargs)

    # Save run info
    run_mgr.save_run_info({
        "model_params": model.num_parameters,
        "train_samples": len(train_spectra),
        "val_samples": len(val_spectra),
        "batch_size": config['pretrain']['batch_size'],
        "epochs": config['pretrain']['epochs'],
        "mask_ratio": config['pretrain']['mask_ratio'],
        "contiguous_masking": contiguous_masking,
        "block_sizes": block_sizes,
        "peak_bias_enabled": peak_bias_enabled,
        "seed": args.seed,
        "status": "running",
    })

    # Train
    print("\nStarting pre-training...")
    print(f"Checkpoints: {run_mgr.checkpoint_dir}")
    print(f"Logs: {run_mgr.log_dir}")

    trainer.fit(pretrain_module, data_module)

    # Save final raw model weights
    final_path = run_mgr.checkpoint_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")

    # Update run info
    run_mgr.save_run_info({
        "model_params": model.num_parameters,
        "train_samples": len(train_spectra),
        "val_samples": len(val_spectra),
        "batch_size": config['pretrain']['batch_size'],
        "epochs": config['pretrain']['epochs'],
        "mask_ratio": config['pretrain']['mask_ratio'],
        "contiguous_masking": contiguous_masking,
        "block_sizes": block_sizes,
        "peak_bias_enabled": peak_bias_enabled,
        "seed": args.seed,
        "status": "completed",
        "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "final_model": str(final_path),
    })

    print("\n" + "="*60)
    print("Pre-training complete!")
    print("="*60)
    print(f"Run directory: {run_mgr.run_dir}")
    if checkpoint_callback.best_model_score:
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"\nTo evaluate:")
    print(f"  uv run python evaluate_model.py --run_dir {run_mgr.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train LIBS Foundation Model")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to model/training config file')
    parser.add_argument('--libs_data_config', type=str, default=None,
                        help='Path to physics-based LIBS data pipeline config '
                             '(e.g. config/libs_data.yaml). If set, replaces the '
                             'legacy SyntheticLIBSGenerator with the realistic pipeline.')
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Base directory for all runs (default: runs/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Optional experiment name (added to run folder name)')
    parser.add_argument('--save_data', action='store_true',
                        help='Save generated data to run directory')
    parser.add_argument('--contiguous_masking', action='store_true',
                        help='Use contiguous masking (overrides config)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
