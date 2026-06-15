"""
Fine-tuning module for LIBS Foundation Model.

Implements supervised fine-tuning for classification and regression tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Literal
import math
import numpy as np
from scipy.stats import spearmanr

from models.heads import (
    BinnedQuantificationHead,
    ClassificationHead,
    DetectionHead,
    RegressionHead,
    bin_to_concentration,
    concentration_to_bin,
    concentration_to_presence,
)


# Tasks legend:
#   'classification'         — single class label, cross-entropy
#   'quantification'         — concentration vector, MSE (standard regression)
#   'quantification_binned'  — per-element bin CE (upstream-style, 1000-way per element)
#   'detection'              — multi-label element presence/absence (BCE), labels
#                              derived from concentrations vs per-element LODs
#   'regression' / 'both'    — legacy aliases kept for backward compatibility:
#                              'regression' == 'quantification' (sigmoid head),
#                              'both' = classification + regression jointly.
TaskName = Literal[
    'classification',
    'quantification',
    'quantification_binned',
    'detection',
    'regression',
    'both',
]


class LIBSFinetuneModule(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning the LIBS Transformer.

    Supports four downstream tasks (see TaskName). The encoder is shared; each
    task has its own head. Task choice fully determines what targets are read
    from the batch:
        - classification:        batch['label']           (int64, [B])
        - quantification:        batch['concentrations']  (float32, [B, n_elements])
        - quantification_binned: batch['concentrations']  (float32, [B, n_elements])
        - both:                  batch['label'] + batch['concentrations']

    Args:
        encoder: Pre-trained LIBSTransformer encoder
        task: One of TaskName
        n_classes: Number of classes (classification only)
        n_elements: Concentration vector dimension (quantification tasks).
                    Defaults to n_classes for backward compat.
        n_concentration_bins: Bin count for quantification_binned (upstream: 1000)
        freeze_encoder: Whether to freeze encoder weights
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        max_epochs: Maximum epochs
        class_weights: Optional class weights for imbalanced classification
        pool: Pooling strategy for the encoder representation
    """

    def __init__(
        self,
        encoder: nn.Module,
        task: TaskName = 'classification',
        n_classes: int = 5,
        n_elements: Optional[int] = None,
        n_concentration_bins: int = 1000,
        freeze_encoder: bool = False,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        class_weights: Optional[torch.Tensor] = None,
        pool: Literal['cls', 'mean', 'cls_mean'] = 'cls',
        element_names: Optional[list[str]] = None,
        lod: Optional[torch.Tensor] = None,
        detection_pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['encoder', 'class_weights', 'lod', 'detection_pos_weight'])

        self.encoder = encoder
        self.task = task
        self.n_classes = n_classes
        self.n_elements = n_elements if n_elements is not None else n_classes
        self.n_concentration_bins = n_concentration_bins
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.pool = pool
        if element_names is None:
            self.element_names = [f"elem_{i}" for i in range(self.n_elements)]
        else:
            self.element_names = list(element_names)
        if len(self.element_names) != self.n_elements:
            self.element_names = [f"elem_{i}" for i in range(self.n_elements)]

        # Test-only buffers for per-element diagnostics.
        self._test_conc_preds: list[torch.Tensor] = []
        self._test_conc_targets: list[torch.Tensor] = []
        self.test_per_element_metrics: dict[str, dict[str, float]] = {}
        # Test-only buffers for detection (presence) diagnostics.
        self._test_det_probs: list[torch.Tensor] = []
        self._test_det_targets: list[torch.Tensor] = []
        self.test_detection_metrics: dict[str, Any] = {}

        d_model = encoder.d_model
        head_in_dim = 2 * d_model if pool == 'cls_mean' else d_model

        if freeze_encoder:
            self.freeze_encoder()

        # Heads — only built for tasks that need them.
        if task in ('classification', 'both'):
            self.classification_head = ClassificationHead(head_in_dim, n_classes)
        if task in ('quantification', 'regression', 'both'):
            self.regression_head = RegressionHead(head_in_dim, self.n_elements)
        if task == 'quantification_binned':
            self.binned_head = BinnedQuantificationHead(
                d_model=head_in_dim,
                n_elements=self.n_elements,
                n_bins=n_concentration_bins,
            )
        if task == 'detection':
            self.detection_head = DetectionHead(
                d_model=head_in_dim,
                n_elements=self.n_elements,
            )

        # Per-element limit-of-detection thresholds (mass fraction) used to turn
        # concentrations into presence/absence targets for the detection task.
        if lod is not None:
            self.register_buffer('detection_lod', torch.as_tensor(lod, dtype=torch.float32))
        else:
            self.detection_lod = None
        # Optional positive-class weighting for the (often imbalanced) BCE loss.
        if detection_pos_weight is not None:
            self.register_buffer(
                'detection_pos_weight',
                torch.as_tensor(detection_pos_weight, dtype=torch.float32),
            )
        else:
            self.detection_pos_weight = None

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
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
        seq = encoder_output['sequence_embeddings']  # [B, L, d_model]
        kpm = encoder_output.get('key_padding_mask')
        if kpm is not None and kpm.size(1) == seq.size(1) + 1:
            valid = (~kpm[:, 1:]).unsqueeze(-1).float()
            denom = valid.sum(dim=1).clamp(min=1.0)
            mean = (seq * valid).sum(dim=1) / denom
        else:
            mean = seq.mean(dim=1)
        if self.pool == 'mean':
            return mean
        return torch.cat([cls, mean], dim=-1)

    def _encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'tokens' in batch:
            return self.encoder({'tokens': batch['tokens'], 'fit_valid': batch.get('fit_valid')})
        if 'line_features' in batch:
            return self.encoder({'line_features': batch['line_features']})
        return self.encoder(batch['spectrum'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dict with 'spectrum' or 'line_features'

        Returns:
            Dict with at least 'cls_embedding' and 'representation', plus
            task-specific outputs:
              - classification:        'class_logits'       [B, n_classes]
              - quantification:        'concentrations'     [B, n_elements]
              - quantification_binned: 'bin_logits'         [B, n_elements, n_bins]
                                       'concentrations_pred' [B, n_elements] (argmax-decoded)
              - both:                  'class_logits' + 'concentrations'
        """
        if isinstance(batch, dict):
            encoder_output = self._encode(batch)
        else:
            encoder_output = self.encoder(batch)
        representation = self._pool(encoder_output)

        result = {
            'cls_embedding': encoder_output['cls_embedding'],
            'representation': representation,
        }

        if self.task in ('classification', 'both'):
            result['class_logits'] = self.classification_head(representation)
        if self.task in ('quantification', 'regression', 'both'):
            result['concentrations'] = self.regression_head(representation)
        if self.task == 'quantification_binned':
            logits = self.binned_head(representation)        # [B, E, N_BINS]
            result['bin_logits'] = logits
            result['concentrations_pred'] = bin_to_concentration(
                logits.argmax(dim=-1), n_bins=self.n_concentration_bins,
            )
        if self.task == 'detection':
            det_logits = self.detection_head(representation)  # [B, n_elements]
            result['detection_logits'] = det_logits
            result['presence_prob'] = torch.sigmoid(det_logits)
            result['presence_pred'] = (result['presence_prob'] >= 0.5).float()
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

    def compute_binned_loss(
        self,
        bin_logits: torch.Tensor,
        concentrations: torch.Tensor,
    ) -> torch.Tensor:
        """Per-element bin CE loss. Targets are encoded on the fly from float
        concentrations so the dataset can stay task-agnostic.

        Args:
            bin_logits: [B, n_elements, n_bins]
            concentrations: [B, n_elements] float in [0, 1]
        """
        bin_targets = concentration_to_bin(concentrations, n_bins=self.n_concentration_bins)
        # Flatten element axis into the batch axis for CE.
        return nn.functional.cross_entropy(
            bin_logits.reshape(-1, self.n_concentration_bins),
            bin_targets.reshape(-1),
        )

    def compute_binned_metrics(
        self,
        bin_logits: torch.Tensor,
        concentrations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Bin-classification metrics plus decoded-concentration MAE/R²."""
        bin_targets = concentration_to_bin(concentrations, n_bins=self.n_concentration_bins)
        preds = bin_logits.argmax(dim=-1)
        bin_acc = (preds == bin_targets).float().mean()

        # Decoded scalar predictions for direct comparison to MSE baseline.
        decoded = bin_to_concentration(preds, n_bins=self.n_concentration_bins)
        mae = torch.abs(decoded - concentrations).mean()
        ss_res = ((concentrations - decoded) ** 2).sum(dim=0)
        ss_tot = ((concentrations - concentrations.mean(dim=0)) ** 2).sum(dim=0)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).mean()
        return {
            'bin_accuracy': bin_acc,
            'decoded_mae': mae,
            'decoded_r2': r2,
        }
    
    def presence_targets(self, concentrations: torch.Tensor) -> torch.Tensor:
        """Binarize concentrations against the per-element LOD buffer."""
        if self.detection_lod is None:
            raise ValueError(
                "detection task requires `lod` (per-element limits of detection); "
                "none were provided to LIBSFinetuneModule."
            )
        return concentration_to_presence(concentrations, self.detection_lod)

    def compute_detection_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-label BCE-with-logits loss for element presence."""
        pos_weight = self.detection_pos_weight
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight,
        )

    def compute_detection_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Micro-averaged presence metrics over all element decisions."""
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct = (preds == targets).float()
        accuracy = correct.mean()
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        # Exact-match: all elements of a spectrum correct simultaneously.
        exact = (preds == targets).all(dim=1).float().mean()
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact,
        }

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared train/val/test logic, dispatched by self.task.

        Args:
            batch: dict containing 'spectrum' plus task-specific targets
            stage: 'train', 'val', or 'test' (prefix for logged metrics)
        """
        outputs = self(batch)
        total_loss = torch.tensor(0.0, device=self.device)
        on_step = (stage == 'train')
        log_kw = dict(on_step=on_step, on_epoch=True, prog_bar=False, sync_dist=(stage != 'train'))
        results: Dict[str, torch.Tensor] = {}

        # ── Classification ───────────────────────────────────────────────
        if self.task in ('classification', 'both'):
            cls_loss = self.compute_classification_loss(outputs['class_logits'], batch['label'])
            total_loss = total_loss + cls_loss
            self.log(f'{stage}/cls_loss', cls_loss, **log_kw)
            m = self.compute_classification_metrics(outputs['class_logits'], batch['label'])
            self.log(f'{stage}/accuracy', m['accuracy'], **{**log_kw, 'prog_bar': stage != 'train'})
            self.log(f'{stage}/balanced_accuracy', m['balanced_accuracy'], **log_kw)
            results['predictions'] = outputs['class_logits'].argmax(dim=-1)
            results['labels'] = batch['label']

        # ── Standard quantification (regression) ─────────────────────────
        if self.task in ('quantification', 'regression', 'both') and 'concentrations' in batch:
            reg_loss = self.compute_regression_loss(outputs['concentrations'], batch['concentrations'])
            total_loss = total_loss + reg_loss
            self.log(f'{stage}/reg_loss', reg_loss, **log_kw)
            m = self.compute_regression_metrics(outputs['concentrations'], batch['concentrations'])
            self.log(f'{stage}/reg_mae', m['mae'], **{**log_kw, 'prog_bar': stage != 'train'})
            self.log(f'{stage}/reg_r2', m['r2'], **log_kw)
            results['conc_predictions'] = outputs['concentrations']
            results['conc_targets'] = batch['concentrations']

        # ── Binned quantification ────────────────────────────────────────
        if self.task == 'quantification_binned' and 'concentrations' in batch:
            bin_loss = self.compute_binned_loss(outputs['bin_logits'], batch['concentrations'])
            total_loss = total_loss + bin_loss
            self.log(f'{stage}/bin_loss', bin_loss, **log_kw)
            m = self.compute_binned_metrics(outputs['bin_logits'], batch['concentrations'])
            self.log(f'{stage}/bin_accuracy', m['bin_accuracy'], **{**log_kw, 'prog_bar': stage != 'train'})
            self.log(f'{stage}/decoded_mae', m['decoded_mae'], **log_kw)
            self.log(f'{stage}/decoded_r2', m['decoded_r2'], **log_kw)
            results['conc_predictions'] = outputs['concentrations_pred']
            results['conc_targets'] = batch['concentrations']

        # ── Element detection (presence/absence) ─────────────────────────
        if self.task == 'detection' and 'concentrations' in batch:
            det_targets = self.presence_targets(batch['concentrations'])
            det_loss = self.compute_detection_loss(outputs['detection_logits'], det_targets)
            total_loss = total_loss + det_loss
            self.log(f'{stage}/det_loss', det_loss, **log_kw)
            m = self.compute_detection_metrics(outputs['detection_logits'], det_targets)
            self.log(f'{stage}/det_accuracy', m['accuracy'], **{**log_kw, 'prog_bar': stage != 'train'})
            self.log(f'{stage}/det_precision', m['precision'], **log_kw)
            self.log(f'{stage}/det_recall', m['recall'], **log_kw)
            self.log(f'{stage}/det_f1', m['f1'], **{**log_kw, 'prog_bar': stage != 'train'})
            self.log(f'{stage}/det_exact_match', m['exact_match'], **log_kw)
            results['presence_prob'] = outputs['presence_prob']
            results['presence_targets'] = det_targets

        self.log(f'{stage}/loss', total_loss, **{**log_kw, 'prog_bar': True})
        results['loss'] = total_loss
        return results

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')['loss']

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with dedicated `test/*` logging namespace."""
        out = self._step(batch, 'test')
        if self.task == 'quantification_binned':
            pred = out.get('conc_predictions')
            targ = out.get('conc_targets')
            if pred is not None and targ is not None:
                self._test_conc_preds.append(pred.detach().float().cpu())
                self._test_conc_targets.append(targ.detach().float().cpu())
        if self.task == 'detection':
            prob = out.get('presence_prob')
            targ = out.get('presence_targets')
            if prob is not None and targ is not None:
                self._test_det_probs.append(prob.detach().float().cpu())
                self._test_det_targets.append(targ.detach().float().cpu())
        return out

    def on_test_epoch_start(self) -> None:
        self._test_conc_preds = []
        self._test_conc_targets = []
        self.test_per_element_metrics = {}
        self._test_det_probs = []
        self._test_det_targets = []
        self.test_detection_metrics = {}

    @staticmethod
    def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        x_std = x.std()
        y_std = y.std()
        if x_std < 1e-12 or y_std < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        corr = spearmanr(x, y).correlation
        if corr is None or not np.isfinite(corr):
            return 0.0
        return float(corr)

    def on_test_epoch_end(self) -> None:
        if self.task == 'detection':
            self._finalize_detection_test()
            return
        if self.task != 'quantification_binned':
            return
        if not self._test_conc_preds or not self._test_conc_targets:
            return

        preds = torch.cat(self._test_conc_preds, dim=0).numpy()
        targets = torch.cat(self._test_conc_targets, dim=0).numpy()
        n_samples = int(targets.shape[0])
        per_elem: dict[str, dict[str, float]] = {}
        for i in range(self.n_elements):
            name = self.element_names[i] if i < len(self.element_names) else f"elem_{i}"
            y_true = targets[:, i]
            y_pred = preds[:, i]
            mae = float(np.mean(np.abs(y_pred - y_true)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
            pearson = self._safe_pearson(y_true, y_pred)
            spearman = self._safe_spearman(y_true, y_pred)
            per_elem[name] = {
                "mae": mae,
                "r2": r2,
                "pearson": pearson,
                "spearman": spearman,
                "n_samples": float(n_samples),
            }

        # Persist for final run summary / run_info.yaml serialization.
        self.test_per_element_metrics = per_elem

        logger_exp = getattr(self.logger, "experiment", None)
        if logger_exp is None:
            return
        add_scalar = getattr(logger_exp, "add_scalar", None)
        add_hist = getattr(logger_exp, "add_histogram", None)
        if add_scalar is None or add_hist is None:
            return

        step = int(self.current_epoch)
        for i in range(self.n_elements):
            name = self.element_names[i] if i < len(self.element_names) else f"elem_{i}"
            y_true = targets[:, i]
            m = per_elem[name]

            base = f"test/per_element/{name}"
            add_scalar(f"{base}/mae", m["mae"], step)
            add_scalar(f"{base}/r2", m["r2"], step)
            add_scalar(f"{base}/pearson", m["pearson"], step)
            add_scalar(f"{base}/spearman", m["spearman"], step)
            add_hist(f"{base}/target_hist", y_true, step)

        # Keep memory bounded across epochs even if test is called repeatedly.
        self._test_conc_preds = []
        self._test_conc_targets = []
    
    def _finalize_detection_test(self) -> None:
        """Aggregate per-element + overall presence metrics over the test set."""
        if not self._test_det_probs or not self._test_det_targets:
            return
        probs = torch.cat(self._test_det_probs, dim=0).numpy()
        targets = torch.cat(self._test_det_targets, dim=0).numpy()
        preds = (probs >= 0.5).astype(np.float64)
        n_samples = int(targets.shape[0])

        per_elem: dict[str, dict[str, float]] = {}
        for i in range(self.n_elements):
            name = self.element_names[i] if i < len(self.element_names) else f"elem_{i}"
            t = targets[:, i]
            p = preds[:, i]
            tp = float(np.sum(p * t))
            fp = float(np.sum(p * (1 - t)))
            fn = float(np.sum((1 - p) * t))
            tn = float(np.sum((1 - p) * (1 - t)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_elem[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / max(n_samples, 1),
                "support": float(np.sum(t)),       # n present in test set
                "lod": float(self.detection_lod[i]) if self.detection_lod is not None else float("nan"),
                "n_samples": float(n_samples),
            }

        # Macro (unweighted mean over elements) and micro (pooled) summaries.
        macro_f1 = float(np.mean([m["f1"] for m in per_elem.values()]))
        macro_precision = float(np.mean([m["precision"] for m in per_elem.values()]))
        macro_recall = float(np.mean([m["recall"] for m in per_elem.values()]))
        tp = float(np.sum(preds * targets))
        fp = float(np.sum(preds * (1 - targets)))
        fn = float(np.sum((1 - preds) * targets))
        micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) > 0 else 0.0)

        self.test_detection_metrics = {
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "element_accuracy": float(np.mean(preds == targets)),
            "exact_match": float(np.mean((preds == targets).all(axis=1))),
            "per_element": per_elem,
        }

        logger_exp = getattr(self.logger, "experiment", None)
        add_scalar = getattr(logger_exp, "add_scalar", None) if logger_exp else None
        if add_scalar is not None:
            step = int(self.current_epoch)
            for key in ("macro_f1", "micro_f1", "element_accuracy", "exact_match"):
                add_scalar(f"test/detection/{key}", self.test_detection_metrics[key], step)
            for name, m in per_elem.items():
                add_scalar(f"test/detection/per_element/{name}/f1", m["f1"], step)

        self._test_det_probs = []
        self._test_det_targets = []

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate encoder and head parameters for different learning rates
        encoder_params = list(self.encoder.parameters())
        head_params = []
        
        if self.task in ('classification', 'both'):
            head_params.extend(self.classification_head.parameters())
        if self.task in ('quantification', 'regression', 'both'):
            head_params.extend(self.regression_head.parameters())
        if self.task == 'quantification_binned':
            head_params.extend(self.binned_head.parameters())
        if self.task == 'detection':
            head_params.extend(self.detection_head.parameters())
        
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
        
        # Cosine annealing with warmup (guard when warmup_epochs >= max_epochs)
        warmup = max(1, min(int(self.warmup_epochs), int(self.max_epochs) - 1))
        decay_epochs = max(1, int(self.max_epochs) - warmup)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = min(1.0, max(0.0, (epoch - warmup) / decay_epochs))
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
        train_spectra=None,
        train_labels=None,
        val_spectra=None,
        val_labels=None,
        train_concentrations=None,
        val_concentrations=None,
        batch_size: int = 32,
        num_workers: int = 0,
        line_features_path: Optional[str] = None,
        line_tokens_path: Optional[str] = None,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
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
        self.line_features_path = line_features_path
        self.line_tokens_path = line_tokens_path
        self.train_indices = train_indices
        self.val_indices = val_indices
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from data.dataset import (
            LabeledLIBSDataset,
            LineTokenLabeledDataset,
            LineTokensLabeledDataset,
        )
        import numpy as np
        
        if stage == 'fit' or stage is None:
            if self.line_tokens_path:
                self.train_dataset = LineTokensLabeledDataset(
                    self.line_tokens_path,
                    self.train_labels,
                    concentrations=self.train_concentrations,
                    indices=self.train_indices,
                )
                self.val_dataset = LineTokensLabeledDataset(
                    self.line_tokens_path,
                    self.val_labels,
                    concentrations=self.val_concentrations,
                    indices=self.val_indices,
                )
            elif self.line_features_path:
                self.train_dataset = LineTokenLabeledDataset(
                    self.line_features_path,
                    self.train_labels,
                    concentrations=self.train_concentrations,
                    indices=self.train_indices,
                )
                self.val_dataset = LineTokenLabeledDataset(
                    self.line_features_path,
                    self.val_labels,
                    concentrations=self.val_concentrations,
                    indices=self.val_indices,
                )
            else:
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
