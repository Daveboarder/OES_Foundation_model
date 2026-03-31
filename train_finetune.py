"""
Fine-tuning script for LIBS Foundation Model.

Loads a pre-trained model from a run directory, fine-tunes on labeled data,
saves checkpoints and run metadata.

Usage:
    uv run python train_finetune.py --pretrain_run_dir runs/pretrain_... --task both
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
from data.dataset import LabeledLIBSDataset
from models.libs_transformer import LIBSTransformer
from training.finetune import LIBSFinetuneModule, FinetuneDataModule
from utils.run_manager import RunManager


class SaveRawEncoderCallback(Callback):
    """Save raw encoder weights every N epochs for easy mid-training evaluation."""
    def __init__(self, save_path: str, save_every_n_epochs: int = 1):
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(pl_module.encoder.state_dict(), self.save_path)
            info_path = str(self.save_path).replace('.pt', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"epoch: {trainer.current_epoch + 1}\n")
                f.write(f"global_step: {trainer.global_step}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_labeled_data(config: dict, seed: int = 42):
    print("Generating labeled synthetic data...")

    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )

    total_samples = config['data']['synthetic']['labeled_samples']
    spectra, labels, concentrations = generator.generate_dataset(
        n_samples=total_samples, return_labels=True,
    )

    # 70/15/15 split
    train_idx = int(0.7 * total_samples)
    val_idx = int(0.85 * total_samples)

    return {
        'train': (spectra[:train_idx], labels[:train_idx], concentrations[:train_idx]),
        'val': (spectra[train_idx:val_idx], labels[train_idx:val_idx], concentrations[train_idx:val_idx]),
        'test': (spectra[val_idx:], labels[val_idx:], concentrations[val_idx:]),
    }


def load_pretrained_model(config: dict, checkpoint_path: str) -> LIBSTransformer:
    model = LIBSTransformer(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle Lightning checkpoint vs raw weights
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items()
                          if k.startswith('model.')}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained model from {checkpoint_path}")
    return model


def create_fresh_model(config: dict) -> LIBSTransformer:
    model = LIBSTransformer(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
    )
    print(f"Created fresh model with {model.num_parameters:,} parameters")
    return model


def main(args):
    config = load_config(args.config)

    # Resolve pretrained checkpoint
    pretrained_checkpoint = None
    pretrain_run_dir = None

    if args.pretrain_run_dir:
        pretrain_mgr = RunManager.from_existing_run(args.pretrain_run_dir)
        pretrained_checkpoint = pretrain_mgr.get_checkpoint_for_mode("pretrain")
        pretrain_run_dir = str(pretrain_mgr.run_dir)
        if pretrained_checkpoint is None:
            raise ValueError(f"No checkpoint found in pretrain run: {args.pretrain_run_dir}")
        print(f"Using pretrained model from: {pretrain_run_dir}")
        print(f"  checkpoint: {pretrained_checkpoint}")

    # Create run manager
    experiment_name = args.experiment_name or args.task
    run_mgr = RunManager(
        run_type="finetune",
        experiment_name=experiment_name,
        base_dir=args.runs_dir,
        config_path=args.config,
    )

    pl.seed_everything(args.seed)

    data = generate_labeled_data(config, seed=args.seed)
    train_spectra, train_labels, train_conc = data['train']
    val_spectra, val_labels, val_conc = data['val']
    test_spectra, test_labels, test_conc = data['test']

    print(f"Train: {len(train_spectra)}, Val: {len(val_spectra)}, Test: {len(test_spectra)}")

    if pretrained_checkpoint:
        encoder = load_pretrained_model(config, str(pretrained_checkpoint))
    else:
        print("No pre-trained checkpoint provided, creating fresh model...")
        encoder = create_fresh_model(config)

    finetune_module = LIBSFinetuneModule(
        encoder=encoder,
        task=args.task,
        n_classes=config['data']['n_classes'],
        freeze_encoder=args.freeze_encoder,
        learning_rate=config['finetune']['learning_rate'],
        weight_decay=config['finetune']['weight_decay'],
        warmup_epochs=5,
        max_epochs=config['finetune']['epochs'],
    )

    data_module = FinetuneDataModule(
        train_spectra=train_spectra,
        train_labels=train_labels,
        val_spectra=val_spectra,
        val_labels=val_labels,
        train_concentrations=train_conc if args.task in ['regression', 'both'] else None,
        val_concentrations=val_conc if args.task in ['regression', 'both'] else None,
        batch_size=config['finetune']['batch_size'],
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
            tags=wandb_config.get('tags', []) + ['finetune', args.task],
            save_dir=str(run_mgr.log_dir),
            config={
                'model': config['model'],
                'finetune': config['finetune'],
                'task': args.task,
                'pretrain_run': pretrain_run_dir,
                'run_dir': str(run_mgr.run_dir),
            },
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(run_mgr.log_dir),
            name='',
            version='',
        )

    # Monitor metric
    if args.task == 'classification':
        monitor, mon_mode = 'val/accuracy', 'max'
    elif args.task == 'regression':
        monitor, mon_mode = 'val/reg_mae', 'min'
    else:
        monitor, mon_mode = 'val/loss', 'min'

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_mgr.checkpoint_dir),
        filename='best',
        save_top_k=1,
        monitor=monitor,
        mode=mon_mode,
        save_last=True,
        auto_insert_metric_name=False,
    )

    raw_encoder_callback = SaveRawEncoderCallback(
        save_path=str(run_mgr.checkpoint_dir / 'encoder_latest.pt'),
        save_every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor, raw_encoder_callback]
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor=monitor, patience=10, mode=mon_mode, verbose=True,
        ))

    # Run info
    run_mgr.save_run_info({
        "task": args.task,
        "model_params": encoder.num_parameters,
        "train_samples": len(train_spectra),
        "val_samples": len(val_spectra),
        "test_samples": len(test_spectra),
        "freeze_encoder": args.freeze_encoder,
        "pretrain_run": pretrain_run_dir,
        "pretrain_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint else None,
        "seed": args.seed,
        "status": "running",
    })

    # Trainer
    trainer = pl.Trainer(
        accelerator=config['device']['accelerator'],
        devices=1,
        precision=config['device']['precision'],
        max_epochs=config['finetune']['epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    print(f"\nStarting fine-tuning for task: {args.task}")
    print(f"Checkpoints: {run_mgr.checkpoint_dir}")
    print(f"Logs: {run_mgr.log_dir}")

    trainer.fit(finetune_module, data_module)

    # Test
    print("\nEvaluating on test set...")
    test_dataset = LabeledLIBSDataset(
        spectra=test_spectra, labels=test_labels,
        concentrations=test_conc if args.task in ['regression', 'both'] else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['finetune']['batch_size'],
        shuffle=False, num_workers=args.num_workers,
    )

    best_model = LIBSFinetuneModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        encoder=encoder, task=args.task,
        n_classes=config['data']['n_classes'],
    )
    test_results = trainer.test(best_model, dataloaders=test_loader)

    # Save final raw encoder weights
    final_path = run_mgr.checkpoint_dir / 'final_encoder.pt'
    torch.save(encoder.state_dict(), final_path)
    print(f"\nSaved final encoder to {final_path}")

    run_mgr.save_run_info({
        "task": args.task,
        "model_params": encoder.num_parameters,
        "train_samples": len(train_spectra),
        "val_samples": len(val_spectra),
        "test_samples": len(test_spectra),
        "freeze_encoder": args.freeze_encoder,
        "pretrain_run": pretrain_run_dir,
        "seed": args.seed,
        "status": "completed",
        "best_metric": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "final_encoder": str(final_path),
        "test_results": test_results[0] if test_results else None,
    })

    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print(f"Run directory: {run_mgr.run_dir}")
    if checkpoint_callback.best_model_score:
        print(f"Best {monitor}: {checkpoint_callback.best_model_score:.4f}")
    print(f"\nTo evaluate:")
    print(f"  uv run python evaluate_model.py --run_dir {run_mgr.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LIBS Foundation Model")

    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--pretrain_run_dir', type=str, default=None,
                        help='Path to pretrain run directory')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'both'],
                        default='both')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    main(args)
