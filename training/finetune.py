"""
Fine-tuning module for LIBS Foundation Model.

Implements supervised fine-tuning for classification and regression tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Literal
import math
import warnings
import numpy as np
from scipy.stats import spearmanr

from models.heads import (
    BinnedQuantificationHead,
    CFLineWeightHead,
    CFPlasmaInitHead,
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
#   'cf_quantification'      — calibration-free quantification: concentrations come
#                              from the parameter-free Saha–Boltzmann layer (cf/);
#                              the encoder only feeds a per-line weight head and a
#                              plasma-state init head, trained on synthetic data
#                              (batches with has_plasma_labels == 1) only.
#   'regression' / 'both'    — legacy aliases kept for backward compatibility:
#                              'regression' == 'quantification' (sigmoid head),
#                              'both' = classification + regression jointly.
TaskName = Literal[
    'classification',
    'quantification',
    'quantification_binned',
    'detection',
    'cf_quantification',
    'regression',
    'both',
]

# Elements whose per-element R² is averaged into `cf_r2_major` (those present
# in element_names are used; the rest are ignored).
CF_MAJOR_ELEMENTS = ('Fe', 'C', 'Mn', 'Si', 'Cr', 'Ni', 'Cu', 'Al')

# Keys of `finetune.cf` forwarded verbatim to cf.layer.SahaBoltzmannLayer(cfg=…)
# (only those present in the config; the layer rejects unknown keys).
CF_LAYER_KEYS = ('n_iter', 'ridge', 'prior_T', 'prior_Ne', 'sa_correction',
                 'gamma_nm', 'eps', 'min_area', 'sa_seed_init', 'use_isolation',
                 'min_lines', 'reject_sigma', 'reject_floor')

# Defaults for every `finetune.cf` key the task reads (config overrides win).
CF_CFG_DEFAULTS: dict[str, Any] = {
    # solver (see cf/layer.py)
    'n_iter': 3,
    'ridge': 1e-6,
    'prior_T': 0.1,
    'prior_Ne': 0.1,
    'sa_correction': True,
    'gamma_nm': 0.01,
    'eps': 1e-7,
    'min_area': 0.0,
    'min_lines': 2,
    'reject_sigma': 3.0,
    'reject_floor': 0.15,
    # loss weights (plan: L = L_conc + λ_T·… + λ_Ne·… + λ_Nl·… + λ_w·…)
    'lambda_T': 0.1,
    'lambda_Ne': 0.1,
    'lambda_Nl': 0.1,
    'lambda_w': 0.01,
    # plumbing
    'presence_gate': True,      # gate line weights by the seed detection head (≥ 0.5)
    'pure_physics': False,      # classical weights + solver defaults, no learned heads
    'c0_source': 'binned',      # binned | uniform | truth (truth = debug only)
    'line_dict_path': None,     # line_dict_<h>.h5 with isolation_score / forced
    'weight_hidden': 64,        # CFLineWeightHead hidden width
    'classical': {},            # kwargs for cf.classical.classical_weights (pure_physics)
    # solver defaults used when the init head is bypassed (pure_physics)
    'T0_default': 10000.0,
    'log10_Ne0_default': 17.0,
    'log10_Nl0_default': 16.0,
}


class LIBSFinetuneModule(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning the LIBS Transformer.

    Supports four downstream tasks (see TaskName). The encoder is shared; each
    task has its own head. Task choice fully determines what targets are read
    from the batch:
        - classification:        batch['label']           (int64, [B])
        - quantification:        batch['concentrations']  (float32, [B, n_elements])
        - quantification_binned: batch['concentrations']  (float32, [B, n_elements])
        - detection:             batch['concentrations'] (binarised against LOD)
        - cf_quantification:     batch['concentrations'] + plasma aux targets
                                 (Te, log10_Ne, log10_Nl, is_two_zone,
                                 has_plasma_labels; each float32 [B])
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
        cf_tables: cf.tables.CFTables (cf_quantification only) — element order,
                   ionisation energies, partition functions, LODs.
        cf_cfg: `finetune.cf` config block (see CF_CFG_DEFAULTS).
        seed_binned: frozen LIBSFinetuneModule(task='quantification_binned')
                     providing the closure seed C0 (cf_quantification only).
        seed_detection: frozen LIBSFinetuneModule(task='detection') providing
                     the element-presence gate on line weights.
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
        cf_tables: Any = None,
        cf_cfg: Optional[dict] = None,
        seed_binned: Optional[nn.Module] = None,
        seed_detection: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['encoder', 'class_weights', 'lod', 'detection_pos_weight',
                    'cf_tables', 'cf_cfg', 'seed_binned', 'seed_detection'])

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
        # Test-only buffers for CF plasma-state diagnostics (cf_quantification).
        self._test_cf_buf: dict[str, list[torch.Tensor]] = {}
        self.test_plasma_metrics: dict[str, float] = {}

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

        # Calibration-free quantification: learned line weights + plasma init,
        # frozen seed modules, and the parameter-free Saha–Boltzmann layer.
        self.cf_cfg: Optional[dict] = None
        self.cf_tables = None
        self.seed_binned = seed_binned
        self.seed_detection = seed_detection
        self._classical_fn = None
        self._warned: set[str] = set()
        if task == 'cf_quantification':
            lod = self._init_cf(cf_tables, cf_cfg, d_model, head_in_dim, lod)

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

    # ── cf_quantification: construction ──────────────────────────────────
    def _init_cf(
        self,
        cf_tables: Any,
        cf_cfg: Optional[dict],
        d_model: int,
        head_in_dim: int,
        lod: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Build the CF heads, the Saha–Boltzmann layer and the seed plumbing.

        Returns the LOD vector to register as `detection_lod` (the CF tables'
        LODs when none was passed explicitly).
        """
        if cf_tables is None:
            raise ValueError("task='cf_quantification' requires cf_tables (cf.tables.CFTables)")
        cfg = dict(CF_CFG_DEFAULTS)
        cfg.update({k: v for k, v in (cf_cfg or {}).items()})
        self.cf_cfg = cfg
        self.cf_tables = cf_tables

        table_names = list(getattr(cf_tables, 'element_names', []))
        if table_names and table_names != self.element_names:
            raise ValueError(
                "cf_tables.element_names must match the module's element_names "
                f"(got {table_names[:5]}… vs {self.element_names[:5]}…)"
            )

        self.cf_weight_head = CFLineWeightHead(d_model, hidden=int(cfg['weight_hidden']))
        self.cf_plasma_head = CFPlasmaInitHead(
            head_in_dim,
            init=(float(cfg['T0_default']), float(cfg['log10_Ne0_default']),
                  float(cfg['log10_Nl0_default'])),
        )

        from cf.layer import SahaBoltzmannLayer  # lazy: cf/ is optional for other tasks
        layer_cfg = {k: cfg[k] for k in CF_LAYER_KEYS if k in cfg}
        self.cf_layer = SahaBoltzmannLayer(
            cf_tables, layer_cfg, line_dict_path=cfg.get('line_dict_path'),
        )

        # Atomic number → element column (−1 = not a target element).
        z_to_elem = np.asarray(cf_tables.z_to_elem, dtype=np.int64)
        self.register_buffer('cf_z_to_elem', torch.from_numpy(z_to_elem.copy()))

        # LOD used for censoring in the loss: the layer's buffer if it has one.
        layer_lod = getattr(self.cf_layer, 'lod', None)
        lod_src = layer_lod if layer_lod is not None else cf_tables.lod
        lod_t = torch.as_tensor(np.asarray(
            lod_src.detach().cpu().numpy() if isinstance(lod_src, torch.Tensor) else lod_src,
            dtype=np.float32,
        ))
        self.register_buffer('cf_lod', lod_t)

        major_idx = [self.element_names.index(e) for e in CF_MAJOR_ELEMENTS
                     if e in self.element_names]
        self.register_buffer('cf_major_idx', torch.as_tensor(major_idx, dtype=torch.long))

        # Seeds are frozen and always in eval mode (see train()).
        for seed in (self.seed_binned, self.seed_detection):
            if seed is not None:
                for p in seed.parameters():
                    p.requires_grad_(False)
                seed.eval()

        if cfg['c0_source'] == 'binned' and self.seed_binned is None:
            self._warn_once(
                "cf: c0_source='binned' but no seed_binned module was given — "
                "the closure starts from a uniform C0 instead."
            )
        if cfg['c0_source'] == 'truth':
            self._warn_once("cf: c0_source='truth' uses batch['concentrations'] as C0 (debug only).")
        return lod if lod is not None else lod_t

    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            warnings.warn(msg, stacklevel=2)

    def train(self, mode: bool = True):
        """Standard train()/eval() switch, except the frozen seed modules stay
        in eval mode (no dropout in the C0 / presence seeds while training)."""
        super().train(mode)
        for seed in (self.seed_binned, self.seed_detection):
            if seed is not None:
                seed.eval()
        return self

    # ── cf_quantification: forward helpers ───────────────────────────────
    def _cf_line_gate(self, presence: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Map per-element presence [B, E] onto the lines [B, L] via the atomic
        number channel; lines of non-target elements get gate 0."""
        from data.line_tokenization import F_Z
        n_z = int(self.cf_z_to_elem.numel())
        z = tokens[..., F_Z].round().long().clamp(0, n_z - 1)
        elem = self.cf_z_to_elem[z]                             # [B, L], -1 if not target
        is_target = (elem >= 0).to(presence.dtype)
        gate = torch.gather(presence, 1, elem.clamp(min=0))
        return gate * is_target

    def _cf_classical_weights(self, tokens: torch.Tensor, fit_valid_f: torch.Tensor) -> torch.Tensor:
        """Parameter-free {0, 1} line weights (pure_physics) from
        cf.classical.classical_weights on the raw tokens; the line dictionary's
        isolation_score / forced buffers of the layer are passed through when
        the layer exposes them (cfg['classical'] keys override)."""
        if self._classical_fn is None:
            try:
                from cf.classical import classical_weights
                self._classical_fn = classical_weights
            except ImportError:
                self._warn_once(
                    "cf.classical.classical_weights not importable — pure_physics "
                    "falls back to weights = fit_valid."
                )
                self._classical_fn = False
        if self._classical_fn is False:
            return fit_valid_f
        kwargs = dict(self.cf_cfg.get('classical') or {})
        for key in ('isolation', 'forced'):
            if key in kwargs:
                continue
            buf = getattr(self.cf_layer, 'isolation_score' if key == 'isolation' else key, None)
            if isinstance(buf, torch.Tensor) and buf.numel() == tokens.shape[1]:
                kwargs[key] = buf.detach().cpu().numpy()
        tok_np = tokens.detach().float().cpu().numpy().astype(np.float64)
        valid_np = fit_valid_f.detach().cpu().numpy()
        w_np = np.asarray(self._classical_fn(tok_np, valid_np, **kwargs), dtype=np.float32)
        w = torch.from_numpy(w_np.reshape(tokens.shape[0], tokens.shape[1])).to(tokens.device)
        return w * fit_valid_f

    def _cf_seed_c0(
        self,
        batch: Dict[str, torch.Tensor],
        seed_inputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Closure seed C0 [B, E] (mass fractions, rows sum to 1) or None (uniform)."""
        src = self.cf_cfg['c0_source']
        if src == 'truth':
            c0 = batch.get('concentrations')
            if c0 is None:
                return None
            c0 = c0.float().clamp(min=0.0)
        elif src == 'binned' and self.seed_binned is not None:
            with torch.no_grad():
                c0 = self.seed_binned(seed_inputs)['concentrations_pred'].float().clamp(min=0.0)
        else:
            return None
        s = c0.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(c0, 1.0 / c0.shape[-1])
        return torch.where(s > 0, c0 / s.clamp(min=1e-12), uniform)

    def _cf_forward(
        self,
        batch: Dict[str, torch.Tensor],
        encoder_output: Dict[str, torch.Tensor],
        representation: torch.Tensor,
    ) -> Dict[str, Any]:
        """Line weights → plasma init → seeds → Saha–Boltzmann layer."""
        cfg = self.cf_cfg
        tokens = batch['tokens']
        B, L = tokens.shape[0], tokens.shape[1]
        fit_valid = batch.get('fit_valid')
        if fit_valid is None:
            fit_valid = torch.ones(B, L, dtype=torch.uint8, device=tokens.device)
        fit_valid_f = (fit_valid > 0).to(torch.float32)
        seed_inputs = {'tokens': tokens, 'fit_valid': fit_valid}

        # 1. per-line weights in [0, 1]
        if cfg['pure_physics']:
            w_logits = None
            w = self._cf_classical_weights(tokens, fit_valid_f)
        else:
            w_logits = self.cf_weight_head(encoder_output['sequence_embeddings']).float()
            w = torch.sigmoid(w_logits) * fit_valid_f

        # 2. presence gate from the frozen detection seed
        presence = None
        if self.seed_detection is not None and cfg['presence_gate']:
            with torch.no_grad():
                presence = (self.seed_detection(seed_inputs)['presence_prob'] >= 0.5).float()
            w = w * self._cf_line_gate(presence, tokens)

        # 3. plasma-state initial guess
        if cfg['pure_physics']:
            T0 = torch.full((B,), float(cfg['T0_default']), device=tokens.device)
            log10_Ne0 = torch.full((B,), float(cfg['log10_Ne0_default']), device=tokens.device)
            log10_Nl0 = torch.full((B,), float(cfg['log10_Nl0_default']), device=tokens.device)
        else:
            T0, log10_Ne0, log10_Nl0 = self.cf_plasma_head(representation.float())

        # 4. closure seed
        C0 = self._cf_seed_c0(batch, seed_inputs)

        # 5. physics
        cf_out = self.cf_layer(
            tokens, fit_valid, w, C0=C0, T0=T0, log10_Ne0=log10_Ne0, log10_Nl0=log10_Nl0,
        )
        out: Dict[str, Any] = {
            'concentrations_pred': cf_out['concentrations'],
            'cf_number_fractions': cf_out['number_fractions'],
            'cf_T': cf_out['T'],
            'cf_log10_Ne': cf_out['log10_Ne'],
            'cf_weights': w,
            'cf_censored': cf_out['censored'],
            'cf_intercepts': cf_out['intercepts'],
            'cf_tau0': cf_out['tau0'],
            'cf_n_lines_used': cf_out['n_lines_used'],
            'cf_resid': cf_out.get('resid'),
            'cf_used_mask': cf_out.get('used_mask'),
            'cf_init': {'T0': T0, 'log10_Ne0': log10_Ne0, 'log10_Nl0': log10_Nl0},
            'cf_fit_valid': fit_valid_f,
        }
        if w_logits is not None:
            out['cf_weight_logits'] = w_logits
        if presence is not None:
            out['cf_presence'] = presence
        if C0 is not None:
            out['cf_C0'] = C0
        return out

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
              - detection:             'detection_logits', 'presence_prob', 'presence_pred'
              - cf_quantification:     'concentrations_pred' [B, E] mass fractions from the
                                       Saha–Boltzmann layer, plus 'cf_number_fractions',
                                       'cf_T' [B], 'cf_log10_Ne' [B], 'cf_weights' [B, L],
                                       'cf_censored' [B, E], 'cf_intercepts' [B, E],
                                       'cf_tau0' [B, L], 'cf_n_lines_used' [B, E],
                                       'cf_init' {T0, log10_Ne0, log10_Nl0} each [B]
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
        if self.task == 'cf_quantification':
            if not isinstance(batch, dict) or 'tokens' not in batch:
                raise ValueError("cf_quantification requires line-token batches ('tokens', 'fit_valid')")
            result.update(self._cf_forward(batch, encoder_output, representation))
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

    # ── cf_quantification: loss + metrics ────────────────────────────────
    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of `values` over entries where `mask` > 0 (0 if the mask is empty)."""
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    def compute_cf_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, int]]:
        """Synthetic-only CF training loss and batch metrics.

        Loss (per plan), averaged over samples with has_plasma_labels == 1:
            L = L_conc
              + λ_T  · ((T − Te)/Te)²          [one-zone shots only]
              + λ_Ne · (log10 Ne − log10 Ne_true)²   [one-zone shots only]
              + λ_Nl · (log10 Nl0 − log10 Nl_true)²  [all synthetic shots]
              + λ_w  · (mean_{fit_valid} w − 1)²
            L_conc = mean_e[m_e (ln(C_e+ε) − ln(C_true,e+ε))²]
                   + mean_e[(1−m_e) relu(ln(C_e+ε) − ln LOD_e)²],   m_e = 1[C_true,e ≥ LOD_e]

        Metrics (over every sample with concentrations, measured included):
            cf_log_rmse, cf_within2x (uncensored truth only), cf_r2_major,
            te_mape, ne_log_mae (one-zone synthetic shots only).

        Returns:
            (loss, metrics, counts) — `counts` tells the caller which metrics
            have support in this batch (0 → do not log).
        """
        cfg = self.cf_cfg
        eps = float(cfg['eps'])
        pred = outputs['concentrations_pred'].float()
        B, E = pred.shape
        device = pred.device
        conc_true = batch.get('concentrations')
        has = batch.get('has_plasma_labels')
        h = has.float().reshape(B) if has is not None else torch.zeros(B, device=device)
        is_two = batch.get('is_two_zone')
        one_zone = h * (1.0 - (is_two.float().reshape(B) if is_two is not None else torch.zeros(B, device=device)))

        # Leaf with grad so measured-only batches still return a backward-able 0.
        loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        metrics: Dict[str, torch.Tensor] = {}
        counts = {'conc': 0, 'supervised': int(h.sum().item()), 'one_zone': int(one_zone.sum().item())}

        lod = self.cf_lod.to(device=device, dtype=torch.float32)
        log_lod = torch.log(lod)[None, :]
        T = outputs['cf_T'].float().reshape(B)
        log10_Ne = outputs['cf_log10_Ne'].float().reshape(B)
        log10_Nl0 = outputs['cf_init']['log10_Nl0'].float().reshape(B)
        w = outputs['cf_weights'].float()
        fv = outputs['cf_fit_valid'].float()

        if conc_true is not None:
            conc_true = conc_true.float()
            counts['conc'] = B
            log_pred = torch.log(pred.clamp(min=0.0) + eps)
            log_true = torch.log(conc_true.clamp(min=0.0) + eps)
            m = (conc_true >= lod[None, :]).float()                   # uncensored truth
            d = log_pred - log_true
            term_unc = (m * d ** 2).mean(dim=1)
            term_cen = ((1.0 - m) * torch.relu(log_pred - log_lod) ** 2).mean(dim=1)
            l_conc = self._masked_mean(term_unc + term_cen, h)

            # metrics
            with torch.no_grad():
                n_unc = m.sum()
                if n_unc > 0:
                    metrics['cf_log_rmse'] = torch.sqrt((m * d ** 2).sum() / n_unc)
                    metrics['cf_within2x'] = (m * (d.abs() <= math.log(2.0)).float()).sum() / n_unc
                else:
                    metrics['cf_log_rmse'] = torch.zeros((), device=device)
                    metrics['cf_within2x'] = torch.zeros((), device=device)
                if self.cf_major_idx.numel() > 0 and B > 1:
                    y = conc_true[:, self.cf_major_idx]
                    p = pred[:, self.cf_major_idx]
                    ss_res = ((y - p) ** 2).sum(dim=0)
                    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
                    metrics['cf_r2_major'] = (1.0 - ss_res / (ss_tot + 1e-8)).mean()
                metrics['cf_mean_weight'] = self._masked_mean(
                    (w * fv).sum(dim=1) / fv.sum(dim=1).clamp(min=1.0), torch.ones(B, device=device))
                metrics['cf_n_censored'] = outputs['cf_censored'].float().sum(dim=1).mean()

            if counts['supervised'] > 0:
                loss = loss + l_conc
                metrics['cf_conc_loss'] = l_conc.detach()

        if counts['supervised'] > 0:
            Te = batch['Te'].float().reshape(B)
            log10_Ne_true = batch['log10_Ne'].float().reshape(B)
            log10_Nl_true = batch['log10_Nl'].float().reshape(B)
            if counts['one_zone'] > 0:
                rel_T = (T - Te) / Te.clamp(min=1.0)
                l_T = self._masked_mean(rel_T ** 2, one_zone)
                l_Ne = self._masked_mean((log10_Ne - log10_Ne_true) ** 2, one_zone)
                loss = loss + float(cfg['lambda_T']) * l_T + float(cfg['lambda_Ne']) * l_Ne
                with torch.no_grad():
                    metrics['te_mape'] = self._masked_mean(rel_T.abs(), one_zone)
                    metrics['ne_log_mae'] = self._masked_mean((log10_Ne - log10_Ne_true).abs(), one_zone)
                    metrics['cf_T_loss'] = l_T.detach()
                    metrics['cf_Ne_loss'] = l_Ne.detach()
            l_Nl = self._masked_mean((log10_Nl0 - log10_Nl_true) ** 2, h)
            mean_w = (w * fv).sum(dim=1) / fv.sum(dim=1).clamp(min=1.0)
            l_w = self._masked_mean((mean_w - 1.0) ** 2, h)
            loss = loss + float(cfg['lambda_Nl']) * l_Nl + float(cfg['lambda_w']) * l_w
            with torch.no_grad():
                metrics['cf_Nl_loss'] = l_Nl.detach()
                metrics['cf_w_loss'] = l_w.detach()
                metrics['nl_log_mae'] = self._masked_mean((log10_Nl0 - log10_Nl_true).abs(), h)
        return loss, metrics, counts

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

        # ── Calibration-free quantification ──────────────────────────────
        # Loss only from batches carrying plasma labels (synthetic); measured
        # batches contribute metrics and a zero loss that still has a grad.
        if self.task == 'cf_quantification':
            cf_loss, m, counts = self.compute_cf_loss(outputs, batch)
            total_loss = total_loss + cf_loss
            self.log(f'{stage}/cf_loss', cf_loss, **log_kw)
            if counts['conc'] > 0:
                for key in ('cf_log_rmse', 'cf_within2x', 'cf_r2_major', 'cf_mean_weight',
                            'cf_n_censored', 'cf_conc_loss'):
                    if key in m:
                        self.log(f'{stage}/{key}', m[key],
                                 **{**log_kw, 'prog_bar': key == 'cf_log_rmse' and stage != 'train'})
                results['conc_predictions'] = outputs['concentrations_pred']
                results['conc_targets'] = batch['concentrations']
            for key in ('te_mape', 'ne_log_mae', 'nl_log_mae', 'cf_T_loss', 'cf_Ne_loss',
                        'cf_Nl_loss', 'cf_w_loss'):
                if key in m:
                    self.log(f'{stage}/{key}', m[key], **log_kw)
            results['cf_T'] = outputs['cf_T']
            results['cf_log10_Ne'] = outputs['cf_log10_Ne']
            results['cf_censored'] = outputs['cf_censored']
            results['cf_init'] = outputs['cf_init']

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
        if self.task in ('quantification_binned', 'cf_quantification'):
            pred = out.get('conc_predictions')
            targ = out.get('conc_targets')
            if pred is not None and targ is not None:
                self._test_conc_preds.append(pred.detach().float().cpu())
                self._test_conc_targets.append(targ.detach().float().cpu())
        if self.task == 'cf_quantification':
            B = out['cf_T'].shape[0]
            buf = self._test_cf_buf

            def _push(key: str, value):
                if value is None:
                    value = torch.zeros(B)
                buf.setdefault(key, []).append(value.detach().float().reshape(B, -1).cpu())

            _push('cf_T', out['cf_T'])
            _push('cf_log10_Ne', out['cf_log10_Ne'])
            _push('T0', out['cf_init']['T0'])
            _push('log10_Ne0', out['cf_init']['log10_Ne0'])
            _push('log10_Nl0', out['cf_init']['log10_Nl0'])
            _push('cf_censored', out['cf_censored'])
            for key in ('Te', 'log10_Ne', 'log10_Nl', 'is_two_zone', 'has_plasma_labels'):
                _push(key, batch.get(key))
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
        self._test_cf_buf = {}
        self.test_plasma_metrics = {}

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

    @staticmethod
    def cf_element_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lod: float,
        censored: Optional[np.ndarray] = None,
        eps: float = 1e-7,
    ) -> dict[str, float]:
        """CF per-element diagnostics shared by the test pass and scripts/evaluate_cf.py.

        log_rmse / within_2x are computed over spectra whose true concentration
        is at or above the LOD (uncensored truth); n_censored counts spectra
        the solver flagged as censored (below LOD) when `censored` is given.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        unc = y_true >= lod
        out: dict[str, float] = {
            "n_uncensored_truth": float(unc.sum()),
            "lod": float(lod),
        }
        if unc.any():
            d = np.log(np.clip(y_pred[unc], 0.0, None) + eps) - np.log(y_true[unc] + eps)
            out["log_rmse"] = float(np.sqrt(np.mean(d ** 2)))
            out["within_2x"] = float(np.mean(np.abs(d) <= np.log(2.0)))
        else:
            out["log_rmse"] = float("nan")
            out["within_2x"] = float("nan")
        out["n_censored"] = float(np.sum(censored)) if censored is not None else 0.0
        return out

    def on_test_epoch_end(self) -> None:
        if self.task == 'detection':
            self._finalize_detection_test()
            return
        if self.task not in ('quantification_binned', 'cf_quantification'):
            return
        if self.task == 'cf_quantification':
            self._finalize_cf_plasma_test()
        if not self._test_conc_preds or not self._test_conc_targets:
            return

        preds = torch.cat(self._test_conc_preds, dim=0).numpy()
        targets = torch.cat(self._test_conc_targets, dim=0).numpy()
        n_samples = int(targets.shape[0])
        censored_all = None
        if self.task == 'cf_quantification' and self._test_cf_buf.get('cf_censored'):
            censored_all = torch.cat(self._test_cf_buf['cf_censored'], dim=0).numpy()
            if censored_all.shape[0] != n_samples:
                censored_all = None
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
            # log_rmse / within_2x need a per-element LOD: cf_lod for the CF
            # task, the detection LOD vector for the learned concentration
            # tasks, so the binned seed stays comparable with CF.
            lod_i = None
            if self.task == 'cf_quantification':
                lod_i = float(self.cf_lod[i])
            elif self.detection_lod is not None:
                lod_i = float(self.detection_lod[i])
            if lod_i is not None:
                per_elem[name].update(self.cf_element_metrics(
                    y_true, y_pred, lod_i,
                    censored=None if censored_all is None else censored_all[:, i],
                    eps=float((self.cf_cfg or {}).get('eps', 1e-7)),
                ))

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
            for key in ("log_rmse", "within_2x", "n_censored"):
                if key in m and np.isfinite(m[key]):
                    add_scalar(f"{base}/{key}", m[key], step)
            add_hist(f"{base}/target_hist", y_true, step)
        for key, value in self.test_plasma_metrics.items():
            if isinstance(value, float) and np.isfinite(value):
                add_scalar(f"test/plasma/{key}", value, step)

        # Keep memory bounded across epochs even if test is called repeatedly.
        self._test_conc_preds = []
        self._test_conc_targets = []
        self._test_cf_buf = {}

    def _finalize_cf_plasma_test(self) -> None:
        """T / Ne recovery over one-zone test shots that carry plasma labels
        (two-zone T is ill-defined; measured shots have no labels)."""
        buf = self._test_cf_buf
        if not buf.get('cf_T'):
            self.test_plasma_metrics = {}
            return
        col = {k: torch.cat(v, dim=0).numpy() for k, v in buf.items() if k != 'cf_censored'}
        has = col['has_plasma_labels'][:, 0] > 0.5
        one_zone = has & (col['is_two_zone'][:, 0] < 0.5)
        two_zone = has & ~one_zone
        metrics: dict[str, float] = {
            "n_test": float(has.shape[0]),
            "n_labelled": float(has.sum()),
            "n_one_zone": float(one_zone.sum()),
            "n_two_zone": float(two_zone.sum()),
        }
        if one_zone.any():
            T, Te = col['cf_T'][one_zone, 0], col['Te'][one_zone, 0]
            lne, lne_t = col['cf_log10_Ne'][one_zone, 0], col['log10_Ne'][one_zone, 0]
            metrics["te_mape"] = float(np.mean(np.abs(T - Te) / np.maximum(Te, 1.0)))
            metrics["te_rmse"] = float(np.sqrt(np.mean((T - Te) ** 2)))
            metrics["ne_log_mae"] = float(np.mean(np.abs(lne - lne_t)))
            metrics["ne_log_rmse"] = float(np.sqrt(np.mean((lne - lne_t) ** 2)))
            metrics["t0_mape"] = float(np.mean(np.abs(col['T0'][one_zone, 0] - Te) / np.maximum(Te, 1.0)))
            metrics["ne0_log_mae"] = float(np.mean(np.abs(col['log10_Ne0'][one_zone, 0] - lne_t)))
        if has.any():
            metrics["nl_log_mae"] = float(np.mean(np.abs(
                col['log10_Nl0'][has, 0] - col['log10_Nl'][has, 0])))
        if two_zone.any():
            T, Te = col['cf_T'][two_zone, 0], col['Te'][two_zone, 0]
            metrics["te_mape_two_zone"] = float(np.mean(np.abs(T - Te) / np.maximum(Te, 1.0)))
        self.test_plasma_metrics = metrics
    
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
        if self.task == 'cf_quantification':
            # The Saha–Boltzmann layer has no parameters and the seeds are
            # frozen; only the two CF heads (plus the encoder) train.
            head_params.extend(self.cf_weight_head.parameters())
            head_params.extend(self.cf_plasma_head.parameters())

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
        train_aux / val_aux: Optional dict of per-spectrum float arrays
            (already sliced to the split, e.g. the plasma-state targets from
            data.libs_pipeline.extract_plasma_targets) forwarded to the
            line-token datasets as `aux_targets`; ignored on the intensity path.
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
        train_aux: Optional[Dict[str, np.ndarray]] = None,
        val_aux: Optional[Dict[str, np.ndarray]] = None,
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
        self.train_aux = train_aux
        self.val_aux = val_aux

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
                    aux_targets=self.train_aux,
                )
                self.val_dataset = LineTokensLabeledDataset(
                    self.line_tokens_path,
                    self.val_labels,
                    concentrations=self.val_concentrations,
                    indices=self.val_indices,
                    aux_targets=self.val_aux,
                )
            elif self.line_features_path:
                self.train_dataset = LineTokenLabeledDataset(
                    self.line_features_path,
                    self.train_labels,
                    concentrations=self.train_concentrations,
                    indices=self.train_indices,
                    aux_targets=self.train_aux,
                )
                self.val_dataset = LineTokenLabeledDataset(
                    self.line_features_path,
                    self.val_labels,
                    concentrations=self.val_concentrations,
                    indices=self.val_indices,
                    aux_targets=self.val_aux,
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
