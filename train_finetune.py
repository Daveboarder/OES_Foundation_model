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
from data.libs_pipeline import (
    build_dataset_from_config,
    cluster_compositions,
    extract_finetune_labels,
    get_or_make_splits,
)
from data.line_embedding_pipeline import (
    prepare_line_token_assets,
    prepare_line_tokens_assets,
)
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


def _checkpoint_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if any(k.startswith('model.') for k in state_dict.keys()):
            return {
                k[len('model.'):]: v for k, v in state_dict.items()
                if k.startswith('model.')
            }
        return state_dict
    return checkpoint


def _load_weights_shape_safe(model: LIBSTransformer, state_dict: dict) -> None:
    """Load only tensors with matching shapes (skips mip_head etc. when unused in finetune)."""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"  Skipped {len(skipped)} keys due to shape mismatch (expected for mip_head when finetuning):")
        for k in skipped[:8]:
            print(f"    - {k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(model_sd[k].shape)}")
        if len(skipped) > 8:
            print(f"    ... and {len(skipped) - 8} more")
    if missing:
        n_emb = sum(1 for k in missing if k.startswith('embedding.'))
        n_enc = sum(1 for k in missing if k.startswith('encoder_blocks.'))
        print(f"  Missing keys: {len(missing)} (embedding={n_emb}, encoder={n_enc}, other={len(missing) - n_emb - n_enc})")
        if n_emb > 0 and n_enc > 0:
            raise RuntimeError(
                "Checkpoint architecture does not match the finetune model — "
                "embedding and encoder weights could not be loaded. "
                "Use the same embedding_type and model config as pre-training "
                "(see pretrain run config.yaml / run_info.yaml)."
            )
    if unexpected:
        print(f"  Unexpected keys in checkpoint (ignored): {len(unexpected)}")


def align_config_with_pretrain_run(config: dict, args, pretrain_run_dir: str) -> None:
    """Match model/embedding settings to the pretrain run so checkpoints load correctly."""
    run_dir = Path(pretrain_run_dir)
    run_info_path = run_dir / "run_info.yaml"
    pretrain_cfg_path = run_dir / "config.yaml"
    if not run_info_path.is_file():
        return

    run_info = yaml.safe_load(open(run_info_path))
    pretrain_emb = run_info.get("embedding_type")
    finetune_emb = config.get("model", {}).get("embedding_type", "intensity")

    if pretrain_emb and pretrain_emb != finetune_emb:
        print(
            f"\nAligning with pretrain run: embedding_type {finetune_emb!r} -> {pretrain_emb!r} "
            f"(from {run_info_path})"
        )

    if pretrain_cfg_path.is_file():
        pretrain_cfg = yaml.safe_load(open(pretrain_cfg_path))
        if "model" in pretrain_cfg:
            config["model"].update(pretrain_cfg["model"])

    lec = run_info.get("line_embedding_config")
    if lec:
        if not args.line_embedding_config:
            args.line_embedding_config = lec
            print(f"  Inherited --line_embedding_config {lec}")
        elif Path(lec).resolve() != Path(args.line_embedding_config).resolve():
            print(
                f"  WARNING: --line_embedding_config ({args.line_embedding_config}) "
                f"differs from pretrain ({lec})"
            )


def _generate_legacy_labeled(config: dict, seed: int):
    """Original toy generator with 5 classes and 5-dim concentrations."""
    print("Generating legacy labeled synthetic data...")
    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )
    total = config['data']['synthetic']['labeled_samples']
    spectra, labels, concentrations = generator.generate_dataset(n_samples=total, return_labels=True)
    t = int(0.7 * total); v = int(0.85 * total)
    return {
        'train': (spectra[:t], labels[:t], concentrations[:t]),
        'val':   (spectra[t:v], labels[t:v], concentrations[t:v]),
        'test':  (spectra[v:], labels[v:], concentrations[v:]),
        'element_names': None,
    }


def _generate_libs_pipeline_labeled(config: dict, libs_config_path: str, seed: int):
    """Realistic physics-based labeled data.

    Loads (or generates from cache) the full synthetic dataset, extracts
    per-element concentrations from sample_table, and partitions everything
    into train/val/test using the splits JSON keyed by the cache fingerprint.

    Returns the same dict shape as the legacy path but with concentrations of
    shape [N, n_elements] (60 elements by default) and labels coming from
    KMeans clustering on the concentration vectors.
    """
    print(f"Generating LIBS-pipeline labeled data from {libs_config_path}...")
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})
    # Respect the libs_data.yaml seed unless caller overrides.
    libs_cfg['generation'].setdefault('seed', seed)
    downstream = libs_cfg.get('downstream', {})

    ds = build_dataset_from_config(libs_cfg)
    if len(ds) == 0:
        raise RuntimeError("LIBS pipeline produced no labeled data — check sample matrix / DB.")

    spectra = ds.spectra.astype(np.float32)
    elements = downstream.get('elements_to_predict')  # None = all 60
    concentrations, element_names, sample_type_ids = extract_finetune_labels(
        ds.sample_table, elements=elements,
    )

    # Override n_bins from the actual wavelength array
    actual_n_bins = spectra.shape[1]
    if config['data']['n_bins'] != actual_n_bins:
        print(f"Overriding n_bins: {config['data']['n_bins']} -> {actual_n_bins}")
        config['data']['n_bins'] = actual_n_bins
        config['model']['max_seq_len'] = actual_n_bins + 1

    # Override n_classes (= cluster count), n_elements (= concentration vector dim),
    # and n_concentration_bins (per-element bin count for the binned task).
    n_clusters = downstream.get('n_clusters', 10)
    config['data']['n_classes'] = n_clusters
    config['data']['n_elements'] = len(element_names)
    config['data']['n_concentration_bins'] = downstream.get('n_concentration_bins', 1000)

    cluster_labels = cluster_compositions(concentrations, n_clusters=n_clusters, seed=seed)

    # Shared deterministic split — cached alongside the spectra. Pretrain and
    # finetune will read the same JSON so test set is consistent across phases.
    split_cfg = downstream.get('splits', {})
    splits, splits_path = get_or_make_splits(
        n=len(spectra),
        cache_dir=ds.cache_dir,
        cache_key=ds.cache_key,
        val_fraction=split_cfg.get('val_fraction', 0.15),
        test_fraction=split_cfg.get('test_fraction', 0.15),
        seed=split_cfg.get('seed', seed),
    )
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}  (saved to {splits_path})")
    print(f"Elements: {len(element_names)}  Clusters: {n_clusters}  "
          f"Sample types: {len(np.unique(sample_type_ids))}")

    def pick(idx):
        return spectra[idx], cluster_labels[idx], concentrations[idx]

    return {
        'train': pick(splits['train']),
        'val':   pick(splits['val']),
        'test':  pick(splits['test']),
        'element_names': element_names,
        'splits': splits,
        'libs_dataset': ds,
    }


def generate_labeled_data(config: dict, seed: int = 42, libs_config_path: str | None = None):
    if libs_config_path:
        return _generate_libs_pipeline_labeled(config, libs_config_path, seed)
    return _generate_legacy_labeled(config, seed)


def load_pretrained_model(
    config: dict,
    checkpoint_path: str,
    line_dict_meta: dict | None = None,
    line_token_meta: dict | None = None,
) -> LIBSTransformer:
    emb_type = config['model'].get('embedding_type', 'intensity')
    kwargs = dict(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
        embedding_type=emb_type,
    )
    if emb_type == 'line_token' and line_dict_meta:
        kwargs['n_lines'] = line_dict_meta['n_lines']
        kwargs['line_dict_meta'] = line_dict_meta
        kwargs['n_elements_vocab'] = line_dict_meta.get('n_elements', 53)
    if emb_type == 'line_token_linear' and line_token_meta:
        kwargs['n_lines'] = line_token_meta['n_lines']
        kwargs['line_token_meta'] = line_token_meta
        kwargs['n_mip_target_channels'] = int(
            config['model'].get('n_mip_target_channels', 2)
        )
    model = LIBSTransformer(**kwargs)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = _checkpoint_state_dict(checkpoint)
    _load_weights_shape_safe(model, state_dict)
    print(f"Loaded pre-trained weights from {checkpoint_path}")
    return model


def create_fresh_model(
    config: dict,
    line_dict_meta: dict | None = None,
    line_token_meta: dict | None = None,
) -> LIBSTransformer:
    emb_type = config['model'].get('embedding_type', 'intensity')
    kwargs = dict(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
        embedding_type=emb_type,
    )
    if emb_type == 'line_token' and line_dict_meta:
        kwargs['n_lines'] = line_dict_meta['n_lines']
        kwargs['line_dict_meta'] = line_dict_meta
        kwargs['n_elements_vocab'] = line_dict_meta.get('n_elements', 53)
    if emb_type == 'line_token_linear' and line_token_meta:
        kwargs['n_lines'] = line_token_meta['n_lines']
        kwargs['line_token_meta'] = line_token_meta
        kwargs['n_mip_target_channels'] = int(
            config['model'].get('n_mip_target_channels', 2)
        )
    model = LIBSTransformer(**kwargs)
    print(f"Created fresh model with {model.num_parameters:,} parameters")
    return model


def main(args):
    config = load_config(args.config)

    # Resolve pretrained checkpoint
    pretrained_checkpoint = None
    pretrain_run_dir = None

    if args.pretrain_run_dir:
        pretrain_mgr = RunManager.from_existing_run(args.pretrain_run_dir)
        pretrain_run_dir = str(pretrain_mgr.run_dir)
        align_config_with_pretrain_run(config, args, pretrain_run_dir)
        pretrained_checkpoint = pretrain_mgr.get_checkpoint_for_mode("pretrain")
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

    line_dict_meta = None
    line_token_meta = None
    line_features_path = None
    line_tokens_path = None
    train_indices = val_indices = test_indices = None

    requested_emb = config['model'].get('embedding_type', 'intensity')
    if args.line_embedding_config and requested_emb == 'intensity':
        requested_emb = 'line_token_linear'
        config['model']['embedding_type'] = requested_emb
    use_line_token = requested_emb in ('line_token', 'line_token_linear')
    if use_line_token:
        if not args.line_embedding_config or not args.libs_data_config:
            raise ValueError(
                f"embedding_type={requested_emb!r} finetune requires "
                "--line_embedding_config and --libs_data_config"
            )

    data = generate_labeled_data(config, seed=args.seed, libs_config_path=args.libs_data_config)
    train_spectra, train_labels, train_conc = data['train']
    val_spectra, val_labels, val_conc = data['val']
    test_spectra, test_labels, test_conc = data['test']
    element_names = data.get('element_names')

    if use_line_token and 'libs_dataset' in data:
        ds = data['libs_dataset']
        if requested_emb == 'line_token':
            line_dict_meta = prepare_line_token_assets(
                ds.spectra.astype(np.float32),
                ds.wavelength,
                args.line_embedding_config,
                spectra_cache_key=ds.cache_key,
                verbose=True,
            )
            n_lines = line_dict_meta['n_lines']
            line_features_path = line_dict_meta['line_features_path']
        else:
            line_token_meta = prepare_line_tokens_assets(
                ds.spectra.astype(np.float32),
                ds.wavelength,
                args.line_embedding_config,
                spectra_cache_key=ds.cache_key,
                verbose=True,
            )
            n_lines = line_token_meta['n_lines']
            line_tokens_path = line_token_meta['line_tokens_path']
        config['data']['n_bins'] = n_lines
        config['model']['max_seq_len'] = n_lines + 1
        splits = data['splits']
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']
        print(f"Line tokens: {n_lines} lines per spectrum")

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    if element_names is not None:
        print(f"Predicting {len(element_names)} elements: {element_names[:10]}"
              f"{'...' if len(element_names) > 10 else ''}")

    if pretrained_checkpoint:
        encoder = load_pretrained_model(
            config, str(pretrained_checkpoint),
            line_dict_meta=line_dict_meta,
            line_token_meta=line_token_meta,
        )
    else:
        print("No pre-trained checkpoint provided, creating fresh model...")
        encoder = create_fresh_model(
            config, line_dict_meta=line_dict_meta, line_token_meta=line_token_meta,
        )

    # n_elements: prefer config.data.n_elements (set by the libs pipeline path);
    # fall back to n_classes for the legacy 5-class generator.
    n_elements = config['data'].get('n_elements', config['data']['n_classes'])
    n_concentration_bins = config['data'].get('n_concentration_bins', 1000)

    ft_epochs = int(config['finetune']['epochs'])
    ft_warmup = int(config['finetune'].get('warmup_epochs', min(1, max(0, ft_epochs - 1))))
    ft_warmup = max(1, min(ft_warmup, ft_epochs - 1)) if ft_epochs > 1 else 1

    finetune_module = LIBSFinetuneModule(
        encoder=encoder,
        task=args.task,
        n_classes=config['data']['n_classes'],
        n_elements=n_elements,
        n_concentration_bins=n_concentration_bins,
        freeze_encoder=args.freeze_encoder,
        learning_rate=config['finetune']['learning_rate'],
        weight_decay=config['finetune']['weight_decay'],
        warmup_epochs=ft_warmup,
        max_epochs=ft_epochs,
        pool=args.pool,
        element_names=element_names,
    )

    needs_concentrations = args.task in ('regression', 'quantification', 'quantification_binned', 'both')
    spectra_unused = line_features_path is not None or line_tokens_path is not None
    data_module = FinetuneDataModule(
        train_spectra=train_spectra if not spectra_unused else None,
        train_labels=train_labels,
        val_spectra=val_spectra if not spectra_unused else None,
        val_labels=val_labels,
        train_concentrations=train_conc if needs_concentrations else None,
        val_concentrations=val_conc if needs_concentrations else None,
        batch_size=config['finetune']['batch_size'],
        num_workers=args.num_workers,
        line_features_path=line_features_path,
        line_tokens_path=line_tokens_path,
        train_indices=train_indices,
        val_indices=val_indices,
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

    # Monitor metric — task-specific so the best checkpoint reflects the right goal
    if args.task == 'classification':
        monitor, mon_mode = 'val/accuracy', 'max'
    elif args.task in ('regression', 'quantification'):
        monitor, mon_mode = 'val/reg_mae', 'min'
    elif args.task == 'quantification_binned':
        monitor, mon_mode = 'val/bin_accuracy', 'max'
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
        "pool": args.pool,
        "embedding_type": config['model'].get('embedding_type', 'intensity'),
        "n_elements": n_elements,
        "n_concentration_bins": n_concentration_bins,
        "element_names": element_names,
        "libs_data_config": args.libs_data_config,
        "line_embedding_config": args.line_embedding_config,
        "line_features_path": line_features_path,
        "line_tokens_path": line_tokens_path,
        "model_params": encoder.num_parameters,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
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
    if line_tokens_path:
        from data.dataset import LineTokensLabeledDataset
        test_dataset = LineTokensLabeledDataset(
            line_tokens_path,
            test_labels,
            concentrations=test_conc if needs_concentrations else None,
            indices=test_indices,
        )
    elif line_features_path:
        from data.dataset import LineTokenLabeledDataset
        test_dataset = LineTokenLabeledDataset(
            line_features_path,
            test_labels,
            concentrations=test_conc if needs_concentrations else None,
            indices=test_indices,
        )
    else:
        test_dataset = LabeledLIBSDataset(
            spectra=test_spectra, labels=test_labels,
            concentrations=test_conc if needs_concentrations else None,
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['finetune']['batch_size'],
        shuffle=False, num_workers=args.num_workers,
    )

    best_model = LIBSFinetuneModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        encoder=encoder, task=args.task,
        n_classes=config['data']['n_classes'],
        n_elements=n_elements,
        n_concentration_bins=n_concentration_bins,
        element_names=element_names,
    )
    test_results = trainer.test(best_model, dataloaders=test_loader)
    aggregate_test_results = dict(test_results[0]) if test_results else {}
    per_element_test = getattr(best_model, "test_per_element_metrics", {}) or {}
    if per_element_test:
        aggregate_test_results["per_element"] = per_element_test

    # Save final raw encoder weights
    final_path = run_mgr.checkpoint_dir / 'final_encoder.pt'
    torch.save(encoder.state_dict(), final_path)
    print(f"\nSaved final encoder to {final_path}")

    run_mgr.save_run_info({
        "task": args.task,
        "pool": args.pool,
        "embedding_type": config['model'].get('embedding_type', 'intensity'),
        "n_elements": n_elements,
        "n_concentration_bins": n_concentration_bins,
        "element_names": element_names,
        "libs_data_config": args.libs_data_config,
        "line_embedding_config": args.line_embedding_config,
        "line_features_path": line_features_path,
        "line_tokens_path": line_tokens_path,
        "model_params": encoder.num_parameters,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "freeze_encoder": args.freeze_encoder,
        "pretrain_run": pretrain_run_dir,
        "seed": args.seed,
        "status": "completed",
        "best_metric": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "final_encoder": str(final_path),
        "test_results": aggregate_test_results if aggregate_test_results else None,
    })

    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print(f"Run directory: {run_mgr.run_dir}")
    if checkpoint_callback.best_model_score:
        print(f"Best {monitor}: {checkpoint_callback.best_model_score:.4f}")
    if aggregate_test_results:
        print("\nTest metrics (aggregate):")
        for k in ("test/bin_accuracy", "test/decoded_mae", "test/decoded_r2", "test/loss"):
            if k in aggregate_test_results:
                print(f"  {k}: {float(aggregate_test_results[k]):.6f}")
    if per_element_test:
        print("\nTest metrics (per element):")
        names = element_names if element_names is not None else sorted(per_element_test.keys())
        for name in names:
            if name not in per_element_test:
                continue
            m = per_element_test[name]
            print(
                f"  {name}: mae={m['mae']:.6f}, r2={m['r2']:.6f}, "
                f"pearson={m['pearson']:.6f}, spearman={m['spearman']:.6f}"
            )
    print(f"\nTo evaluate:")
    print(f"  uv run python evaluate_model.py --run_dir {run_mgr.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LIBS Foundation Model")

    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--pretrain_run_dir', type=str, default=None,
                        help='Path to pretrain run directory')
    parser.add_argument('--task', type=str,
                        choices=['classification', 'quantification',
                                 'quantification_binned', 'regression', 'both'],
                        default='both',
                        help='Downstream task: classification (cluster ID), '
                             'quantification (MSE regression on concentrations), '
                             'quantification_binned (per-element bin CE, upstream-style), '
                             'regression/both (legacy, kept for backward compat)')
    parser.add_argument('--libs_data_config', type=str, default=None,
                        help='Path to physics-based LIBS data pipeline config '
                             '(e.g. config/libs_data.yaml). If set, replaces the '
                             'legacy 5-class SyntheticLIBSGenerator.')
    parser.add_argument('--line_embedding_config', type=str, default=None,
                        help='Path to config/line_embedding.yaml for line-as-token mode.')
    parser.add_argument('--pool', type=str, choices=['cls', 'mean', 'cls_mean'],
                        default='cls',
                        help='How to pool encoder outputs for the heads: '
                             'cls (CLS token only), mean (mean over bins), '
                             'cls_mean (concat of both, head input is 2*d_model)')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    main(args)
