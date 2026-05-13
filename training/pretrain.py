"""
Pre-training module for LIBS Foundation Model.

Implements self-supervised pre-training using masked intensity prediction (MIP).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Optional
import math


class LIBSPretrainModule(pl.LightningModule):
    """
    PyTorch Lightning module for pre-training LIBS Transformer.
    
    Uses masked intensity prediction as the self-supervised objective.
    
    Args:
        model: LIBSTransformer model
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs
        max_epochs: Maximum number of training epochs
        min_lr: Minimum learning rate for scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        
        # Loss function
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, mask=mask)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked intensity prediction loss.
        
        Args:
            predictions: Predicted intensities [batch_size, n_bins]
            targets: Target intensities [batch_size, n_bins]
            mask: Boolean mask [batch_size, n_bins]
            
        Returns:
            MSE loss over masked positions
        """
        # Select only masked positions
        masked_preds = predictions[mask]
        masked_targets = targets[mask]
        
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        return self.mse_loss(masked_preds, masked_targets)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Dictionary with 'input', 'target', 'mask', 'mask_type'
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Only replace mask-token positions (type 1) with learnable mask embedding.
        # Type 2 (random value) and type 3 (unchanged) keep their projected embeddings,
        # preserving the BERT-style 80/10/10 strategy.
        embedding_mask = (batch['mask_type'] == 1)
        outputs = self.model(batch['input'], mask=embedding_mask)

        # Compute loss on ALL masked positions (all 15%), not just mask-token ones
        loss = self.compute_loss(
            predictions=outputs['mip_predictions'],
            targets=batch['target'],
            mask=batch['mask'],
        )
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Dictionary with 'input', 'target', 'mask', 'mask_type'
            batch_idx: Batch index

        Returns:
            Dictionary with loss and predictions
        """
        # Same BERT-style masking as training: only type-1 positions get mask embedding
        embedding_mask = (batch['mask_type'] == 1)
        outputs = self.model(batch['input'], mask=embedding_mask)

        # Loss on ALL masked positions
        loss = self.compute_loss(
            predictions=outputs['mip_predictions'],
            targets=batch['target'],
            mask=batch['mask'],
        )
        
        # Compute additional metrics
        with torch.no_grad():
            masked_preds = outputs['mip_predictions'][batch['mask']]
            masked_targets = batch['target'][batch['mask']]
            
            if masked_preds.numel() > 0:
                mae = torch.abs(masked_preds - masked_targets).mean()
                # R-squared (coefficient of determination)
                ss_res = ((masked_targets - masked_preds) ** 2).sum()
                ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
            else:
                mae = torch.tensor(0.0)
                r2 = torch.tensor(0.0)
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mae', mae, on_epoch=True, sync_dist=True)
        self.log('val/r2', r2, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'predictions': outputs['mip_predictions']}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                min_factor = self.min_lr / self.learning_rate
                return min_factor + (1 - min_factor) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log current epoch
        self.log('epoch', float(self.current_epoch))


class PretrainDataModule(pl.LightningDataModule):
    """
    DataModule for pre-training.
    
    Handles data loading for self-supervised pre-training with advanced masking.
    
    Args:
        train_spectra: Training spectra array
        val_spectra: Validation spectra array
        batch_size: Batch size
        mask_ratio: Fraction of positions to mask
        contiguous_masking: Whether to use contiguous block masking
        block_sizes: List of possible block sizes (e.g., [25, 50, 100, 150])
        peak_bias_enabled: Whether to bias masking toward peaks
        peak_bias_ratio: Fraction of masks that should cover peaks
        peak_threshold: Intensity threshold for peak detection
        num_workers: Number of data loading workers
        
        # Legacy parameter (backward compatibility)
        contiguous_mask_size: Single block size (overridden by block_sizes if provided)
    """
    
    def __init__(
        self,
        train_spectra,
        val_spectra,
        batch_size: int = 64,
        mask_ratio: float = 0.15,
        contiguous_masking: bool = False,
        block_sizes: Optional[list] = None,
        peak_bias_enabled: bool = False,
        peak_bias_ratio: float = 0.5,
        peak_threshold: float = 0.2,
        num_workers: int = 0,
        # Legacy
        contiguous_mask_size: int = 50,
    ):
        super().__init__()
        self.train_spectra = train_spectra
        self.val_spectra = val_spectra
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.contiguous_masking = contiguous_masking
        
        # Handle block sizes (new) vs contiguous_mask_size (legacy)
        if block_sizes is not None:
            self.block_sizes = block_sizes
        else:
            self.block_sizes = [contiguous_mask_size]
        
        self.peak_bias_enabled = peak_bias_enabled
        self.peak_bias_ratio = peak_bias_ratio
        self.peak_threshold = peak_threshold
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from data.dataset import MaskedLIBSDataset
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MaskedLIBSDataset(
                spectra=self.train_spectra,
                mask_ratio=self.mask_ratio,
                contiguous_masking=self.contiguous_masking,
                block_sizes=self.block_sizes,
                peak_bias_enabled=self.peak_bias_enabled,
                peak_bias_ratio=self.peak_bias_ratio,
                peak_threshold=self.peak_threshold,
            )
            self.val_dataset = MaskedLIBSDataset(
                spectra=self.val_spectra,
                mask_ratio=self.mask_ratio,
                contiguous_masking=self.contiguous_masking,
                block_sizes=self.block_sizes,
                peak_bias_enabled=self.peak_bias_enabled,
                peak_bias_ratio=self.peak_bias_ratio,
                peak_threshold=self.peak_threshold,
            )
            
            # Log masking statistics
            stats = self.train_dataset.get_masking_stats(n_samples=100)
            print(f"\nMasking Statistics (from 100 samples):")
            print(f"  Avg masked bins: {stats['avg_masked_bins']:.1f}")
            print(f"  Avg masked ratio: {stats['avg_masked_ratio']:.2%}")
            print(f"  Peak coverage: {stats['peak_coverage']:.2%}")
            print(f"  Avg peak bins: {stats['avg_peak_bins']:.1f}")
    
    def train_dataloader(self):
        from data.dataset import libs_worker_init_fn
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=libs_worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        from data.dataset import libs_worker_init_fn
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=libs_worker_init_fn if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from models.libs_transformer import LIBSTransformer
    from data.synthetic_generator import SyntheticLIBSGenerator
    from data.dataset import MaskedLIBSDataset
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticLIBSGenerator(seed=42)
    spectra, _, _ = generator.generate_dataset(n_samples=1000)
    
    # Split data
    train_spectra = spectra[:800]
    val_spectra = spectra[800:]
    
    # Create model
    print("Creating model...")
    model = LIBSTransformer(
        n_bins=2048,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    )
    
    # Create Lightning module
    pretrain_module = LIBSPretrainModule(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_epochs=2,
        max_epochs=10,
    )
    
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Create data module
    data_module = PretrainDataModule(
        train_spectra=train_spectra,
        val_spectra=val_spectra,
        batch_size=32,
        mask_ratio=0.15,
    )
    
    # Test a single batch
    print("\nTesting single batch...")
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    
    # Test training step
    loss = pretrain_module.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
