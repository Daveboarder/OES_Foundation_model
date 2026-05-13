"""
Fine-tuning module for LIBS Foundation Model.

Implements supervised fine-tuning for classification and regression tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Literal
import math

from models.heads import ClassificationHead, RegressionHead


class LIBSFinetuneModule(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning LIBS Transformer.
    
    Supports classification, regression, or both tasks.
    
    Args:
        encoder: Pre-trained LIBSTransformer encoder
        task: Task type ('classification', 'regression', or 'both')
        n_classes: Number of classes
        freeze_encoder: Whether to freeze encoder weights
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        max_epochs: Maximum epochs
        class_weights: Optional class weights for imbalanced classification
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        task: Literal['classification', 'regression', 'both'] = 'both',
        n_classes: int = 5,
        freeze_encoder: bool = False,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        class_weights: Optional[torch.Tensor] = None,
        pool: Literal['cls', 'mean', 'cls_mean'] = 'cls',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'class_weights'])

        self.encoder = encoder
        self.task = task
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.pool = pool

        d_model = encoder.d_model
        # cls_mean concats CLS and mean-pool, doubling the head input dim
        head_in_dim = 2 * d_model if pool == 'cls_mean' else d_model

        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder()

        # Task heads
        if task in ['classification', 'both']:
            self.classification_head = ClassificationHead(head_in_dim, n_classes)
            if class_weights is not None:
                self.register_buffer('class_weights', class_weights)
            else:
                self.class_weights = None

        if task in ['regression', 'both']:
            self.regression_head = RegressionHead(head_in_dim, n_classes)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
    
    def freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def _pool(self, encoder_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the downstream representation from encoder outputs.

        - 'cls': CLS token only [B, d_model]
        - 'mean': mean over sequence positions (excluding CLS) [B, d_model]
        - 'cls_mean': concat of CLS and mean-pool [B, 2*d_model]
        """
        cls = encoder_output['cls_embedding']
        if self.pool == 'cls':
            return cls
        seq = encoder_output['sequence_embeddings']  # [B, n_bins, d_model]
        mean = seq.mean(dim=1)
        if self.pool == 'mean':
            return mean
        return torch.cat([cls, mean], dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input spectrum [batch_size, n_bins]

        Returns:
            Dictionary with predictions
        """
        encoder_output = self.encoder(x)
        representation = self._pool(encoder_output)

        result = {
            'cls_embedding': encoder_output['cls_embedding'],
            'representation': representation,
        }

        if self.task in ['classification', 'both']:
            result['class_logits'] = self.classification_head(representation)

        if self.task in ['regression', 'both']:
            result['concentrations'] = self.regression_head(representation)

        return result
    
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss."""
        return self.ce_loss(logits, labels)
    
    def compute_regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regression loss."""
        return self.mse_loss(predictions, targets)
    
    def compute_classification_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute classification metrics."""
        preds = logits.argmax(dim=-1)
        
        # Accuracy
        correct = (preds == labels).float()
        accuracy = correct.mean()
        
        # Per-class accuracy
        per_class_acc = []
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc.append(correct[mask].mean())
        
        balanced_acc = torch.stack(per_class_acc).mean() if per_class_acc else accuracy
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
        }
    
    def compute_regression_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute regression metrics."""
        # MSE
        mse = self.mse_loss(predictions, targets)
        
        # MAE
        mae = torch.abs(predictions - targets).mean()
        
        # R-squared (per output, then averaged)
        ss_res = ((targets - predictions) ** 2).sum(dim=0)
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).mean()
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(batch['spectrum'])
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Classification loss
        if self.task in ['classification', 'both']:
            cls_loss = self.compute_classification_loss(
                outputs['class_logits'],
                batch['label'],
            )
            total_loss = total_loss + cls_loss
            self.log('train/cls_loss', cls_loss, on_step=True, on_epoch=True)
            
            # Metrics
            metrics = self.compute_classification_metrics(
                outputs['class_logits'],
                batch['label'],
            )
            self.log('train/accuracy', metrics['accuracy'], on_step=True, on_epoch=True)
        
        # Regression loss
        if self.task in ['regression', 'both'] and 'concentrations' in batch:
            reg_loss = self.compute_regression_loss(
                outputs['concentrations'],
                batch['concentrations'],
            )
            total_loss = total_loss + reg_loss
            self.log('train/reg_loss', reg_loss, on_step=True, on_epoch=True)
            
            # Metrics
            metrics = self.compute_regression_metrics(
                outputs['concentrations'],
                batch['concentrations'],
            )
            self.log('train/reg_mae', metrics['mae'], on_step=True, on_epoch=True)
        
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self(batch['spectrum'])
        
        total_loss = torch.tensor(0.0, device=self.device)
        results = {}
        
        # Classification
        if self.task in ['classification', 'both']:
            cls_loss = self.compute_classification_loss(
                outputs['class_logits'],
                batch['label'],
            )
            total_loss = total_loss + cls_loss
            self.log('val/cls_loss', cls_loss, on_epoch=True, sync_dist=True)
            
            metrics = self.compute_classification_metrics(
                outputs['class_logits'],
                batch['label'],
            )
            self.log('val/accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/balanced_accuracy', metrics['balanced_accuracy'], on_epoch=True, sync_dist=True)
            
            results['predictions'] = outputs['class_logits'].argmax(dim=-1)
            results['labels'] = batch['label']
        
        # Regression
        if self.task in ['regression', 'both'] and 'concentrations' in batch:
            reg_loss = self.compute_regression_loss(
                outputs['concentrations'],
                batch['concentrations'],
            )
            total_loss = total_loss + reg_loss
            self.log('val/reg_loss', reg_loss, on_epoch=True, sync_dist=True)
            
            metrics = self.compute_regression_metrics(
                outputs['concentrations'],
                batch['concentrations'],
            )
            self.log('val/reg_mae', metrics['mae'], on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/reg_r2', metrics['r2'], on_epoch=True, sync_dist=True)
            
            results['conc_predictions'] = outputs['concentrations']
            results['conc_targets'] = batch['concentrations']
        
        self.log('val/loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        results['loss'] = total_loss
        
        return results
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step (same as validation)."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate encoder and head parameters for different learning rates
        encoder_params = list(self.encoder.parameters())
        head_params = []
        
        if self.task in ['classification', 'both']:
            head_params.extend(list(self.classification_head.parameters()))
        if self.task in ['regression', 'both']:
            head_params.extend(list(self.regression_head.parameters()))
        
        # Use lower learning rate for encoder
        param_groups = [
            {'params': encoder_params, 'lr': self.learning_rate * 0.1},
            {'params': head_params, 'lr': self.learning_rate},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }


class FinetuneDataModule(pl.LightningDataModule):
    """
    DataModule for fine-tuning.
    
    Args:
        train_spectra: Training spectra
        train_labels: Training labels
        val_spectra: Validation spectra
        val_labels: Validation labels
        train_concentrations: Optional training concentrations
        val_concentrations: Optional validation concentrations
        batch_size: Batch size
        num_workers: Number of workers
    """
    
    def __init__(
        self,
        train_spectra,
        train_labels,
        val_spectra,
        val_labels,
        train_concentrations=None,
        val_concentrations=None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_spectra = train_spectra
        self.train_labels = train_labels
        self.val_spectra = val_spectra
        self.val_labels = val_labels
        self.train_concentrations = train_concentrations
        self.val_concentrations = val_concentrations
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from data.dataset import LabeledLIBSDataset
        
        if stage == 'fit' or stage is None:
            self.train_dataset = LabeledLIBSDataset(
                spectra=self.train_spectra,
                labels=self.train_labels,
                concentrations=self.train_concentrations,
            )
            self.val_dataset = LabeledLIBSDataset(
                spectra=self.val_spectra,
                labels=self.val_labels,
                concentrations=self.val_concentrations,
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from models.libs_transformer import LIBSTransformer
    from data.synthetic_generator import SyntheticLIBSGenerator
    
    # Generate data
    print("Generating data...")
    generator = SyntheticLIBSGenerator(seed=42)
    spectra, labels, concentrations = generator.generate_dataset(n_samples=1000)
    
    # Split
    train_idx = 800
    train_spectra, val_spectra = spectra[:train_idx], spectra[train_idx:]
    train_labels, val_labels = labels[:train_idx], labels[train_idx:]
    train_conc, val_conc = concentrations[:train_idx], concentrations[train_idx:]
    
    # Create encoder
    print("Creating model...")
    encoder = LIBSTransformer(
        n_bins=2048,
        d_model=256,
        n_heads=8,
        n_layers=6,
    )
    
    # Create fine-tune module
    finetune_module = LIBSFinetuneModule(
        encoder=encoder,
        task='both',
        n_classes=5,
        freeze_encoder=False,
        learning_rate=5e-5,
    )
    
    # Create data module
    data_module = FinetuneDataModule(
        train_spectra=train_spectra,
        train_labels=train_labels,
        val_spectra=val_spectra,
        val_labels=val_labels,
        train_concentrations=train_conc,
        val_concentrations=val_conc,
        batch_size=32,
    )
    
    # Test batch
    print("\nTesting batch...")
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Spectrum shape: {batch['spectrum'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Concentrations shape: {batch['concentrations'].shape}")
    
    # Test training step
    loss = finetune_module.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
