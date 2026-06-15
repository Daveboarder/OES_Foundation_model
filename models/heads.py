"""
Task-specific heads for LIBS Foundation Model.

Includes heads for:
- Classification (material identification)
- Regression (concentration prediction)        — "standard" quantification
- Binned quantification (per-element bin CE)   — inherited from
  Daveboarder/Element_Identification (train_nn_autotransformer.py).
- Masked intensity prediction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for material identification.
    
    Takes the CLS embedding and produces class logits.
    
    Args:
        d_model: Input dimension (model dimension)
        n_classes: Number of output classes
        hidden_dim: Optional hidden layer dimension (defaults to d_model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cls_embedding: CLS token embedding [batch_size, d_model]
            
        Returns:
            Class logits [batch_size, n_classes]
        """
        return self.classifier(cls_embedding)


class RegressionHead(nn.Module):
    """
    Regression head for concentration prediction.
    
    Takes the CLS embedding and produces concentration values for each class.
    
    Args:
        d_model: Input dimension (model dimension)
        n_outputs: Number of output values (number of classes/elements)
        hidden_dim: Optional hidden layer dimension
        dropout: Dropout rate
        output_activation: Activation for output ('sigmoid', 'softmax', or None)
    """
    
    def __init__(
        self,
        d_model: int,
        n_outputs: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_activation: Optional[str] = 'sigmoid',
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or d_model
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )
        
        if output_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Identity()
    
    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cls_embedding: CLS token embedding [batch_size, d_model]
            
        Returns:
            Concentration predictions [batch_size, n_outputs]
        """
        x = self.regressor(cls_embedding)
        return self.activation(x)


def concentration_to_bin(concentration, n_bins: int = 1000):
    """Map concentration in [0, 1] to an integer bin index in [0, n_bins-1].

    Mirrors upstream Daveboarder/Element_Identification convention:
        c=0.0 -> 0,   c=0.75 -> 749,   c=1.0 -> n_bins-1
    Works on numpy arrays or torch tensors.
    """
    if isinstance(concentration, torch.Tensor):
        idx = torch.round(concentration * (n_bins - 1)).long()
        return idx.clamp_(0, n_bins - 1)
    arr = np.asarray(concentration, dtype=np.float64)
    return np.clip(np.round(arr * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)


def bin_to_concentration(bin_idx, n_bins: int = 1000):
    """Inverse of concentration_to_bin. Returns float in [0, 1]."""
    if isinstance(bin_idx, torch.Tensor):
        return bin_idx.to(torch.float32) / (n_bins - 1)
    return np.asarray(bin_idx, dtype=np.float64) / (n_bins - 1)


class BinnedQuantificationHead(nn.Module):
    """Per-element bin-classification head for concentration prediction.

    Architecture: one small 2-layer MLP branch per element, each producing
    `n_bins` logits. Loss is cross-entropy applied independently per element
    over the bin index target. At inference, argmax → bin_to_concentration.

    Rationale (from upstream): concentrations span several orders of magnitude
    (Fe ~0.95 vs trace ~1e-5), so a discrete classification objective gives a
    more uniform gradient signal than MSE.

    Args:
        d_model: encoder representation dim
        n_elements: number of element concentrations to predict
        n_bins: number of discrete concentration bins (default 1000 = upstream)
        hidden: per-branch hidden width
        dropout: branch dropout
    """

    def __init__(
        self,
        d_model: int,
        n_elements: int,
        n_bins: int = 1000,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_bins = n_bins
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_bins),
            )
            for _ in range(n_elements)
        ])

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representation: [B, d_model] pooled encoder output
        Returns:
            logits: [B, n_elements, n_bins]
        """
        # Stack along element axis; each branch independently consumes the
        # pooled representation. (Could be fused into one big linear, but
        # per-element branches match upstream and isolate gradients.)
        per_element = [branch(representation) for branch in self.branches]
        return torch.stack(per_element, dim=1)


def concentration_to_presence(concentration, lod):
    """Binarize concentrations against per-element limits of detection.

    Args:
        concentration: [..., n_elements] mass fractions in [0, 1].
        lod: [n_elements] per-element limit-of-detection thresholds.

    Returns:
        Float {0, 1} presence labels of the same type/shape as the input:
        1 where concentration >= lod (present), 0 otherwise (absent).
    """
    if isinstance(concentration, torch.Tensor):
        lod_t = lod if isinstance(lod, torch.Tensor) else torch.as_tensor(
            lod, dtype=concentration.dtype, device=concentration.device)
        return (concentration >= lod_t).to(concentration.dtype)
    arr = np.asarray(concentration, dtype=np.float64)
    return (arr >= np.asarray(lod, dtype=np.float64)).astype(np.float32)


class DetectionHead(nn.Module):
    """Multi-label presence/absence head for element detection.

    Produces one logit per element; train with ``BCEWithLogitsLoss`` (no
    terminal Sigmoid). Each element is an independent binary decision
    (present vs. absent), unlike ``ClassificationHead`` which is mutually
    exclusive across classes.

    Args:
        d_model: encoder representation dim
        n_elements: number of elements to score for presence
        hidden_dim: hidden layer width (defaults to d_model)
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_elements: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.n_elements = n_elements
        self.detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_elements),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representation: [B, d_model] pooled encoder output
        Returns:
            Presence logits [B, n_elements] (raw; apply sigmoid for probs).
        """
        return self.detector(representation)


class MaskedBinIntensityHead(nn.Module):
    """
    Classification head for masked bin-intensity prediction (MIP).

    Outputs raw logits per bin; use with ``nn.CrossEntropyLoss`` (no Softmax).

    Args:
        d_model: Transformer hidden size.
        num_intensity_bins: Number of discrete intensity classes.
    """

    def __init__(self, d_model: int, num_intensity_bins: int = 256):
        super().__init__()
        self.num_intensity_bins = num_intensity_bins
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_intensity_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sequence embeddings ``[B, Seq_Len, d_model]``.

        Returns:
            Logits ``[B, Seq_Len, num_intensity_bins]``.
        """
        return self.head(x)


class MaskedLineFeatureHead(nn.Module):
    """
    Dual classification heads for Voigt line tokens (max intensity + FWHM).

    Two decoupled stacks on the shared trunk; no terminal Softmax.

    Args:
        d_model: Transformer hidden size.
        num_intensity_bins: Classes for max_intensity (log-spaced targets).
        num_fwhm_bins: Classes for FWHM (uniform-spaced targets).
    """

    def __init__(
        self,
        d_model: int,
        num_intensity_bins: int = 256,
        num_fwhm_bins: int = 100,
    ):
        super().__init__()
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_intensity_bins),
        )
        self.fwhm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_fwhm_bins),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Sequence embeddings ``[B, Seq_Len, d_model]``.

        Returns:
            ``(intensity_logits, fwhm_logits)`` each ``[B, Seq_Len, num_*_bins]``.
        """
        return self.intensity_head(x), self.fwhm_head(x)


class MaskedPredictionHead(nn.Module):
    """
    Head for masked intensity prediction.
    
    Takes sequence embeddings and predicts original intensity values.
    
    Args:
        d_model: Input dimension
        hidden_dim: Optional hidden layer dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or d_model
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequence_embeddings: Sequence embeddings [batch_size, n_bins, d_model]
            
        Returns:
            Intensity predictions [batch_size, n_bins]
        """
        return self.predictor(sequence_embeddings).squeeze(-1)


class MultiTaskHead(nn.Module):
    """
    Multi-task head combining classification and regression.
    
    Useful for joint training on both tasks.
    
    Args:
        d_model: Input dimension
        n_classes: Number of classes for classification
        n_outputs: Number of outputs for regression
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_classes: int = 5,
        n_outputs: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.classification_head = ClassificationHead(
            d_model=d_model,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        self.regression_head = RegressionHead(
            d_model=d_model,
            n_outputs=n_outputs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_activation='sigmoid',
        )
    
    def forward(self, cls_embedding: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            cls_embedding: CLS token embedding [batch_size, d_model]
            
        Returns:
            Dictionary with 'class_logits' and 'concentrations'
        """
        return {
            'class_logits': self.classification_head(cls_embedding),
            'concentrations': self.regression_head(cls_embedding),
        }


class LIBSModelWithHeads(nn.Module):
    """
    Complete LIBS model with encoder and task heads.
    
    Combines the transformer encoder with classification and/or regression heads.
    
    Args:
        encoder: Pre-trained LIBSTransformer encoder
        n_classes: Number of classes for classification
        use_classification: Whether to include classification head
        use_regression: Whether to include regression head
        freeze_encoder: Whether to freeze encoder weights
        dropout: Dropout rate for heads
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int = 5,
        use_classification: bool = True,
        use_regression: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = encoder
        d_model = encoder.d_model
        
        self.use_classification = use_classification
        self.use_regression = use_regression
        
        if use_classification:
            self.classification_head = ClassificationHead(
                d_model=d_model,
                n_classes=n_classes,
                dropout=dropout,
            )
        
        if use_regression:
            self.regression_head = RegressionHead(
                d_model=d_model,
                n_outputs=n_classes,
                dropout=dropout,
            )
        
        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input spectrum [batch_size, n_bins]
            
        Returns:
            Dictionary with encoder outputs and task predictions
        """
        # Get encoder outputs
        encoder_output = self.encoder(x)
        cls_embedding = encoder_output['cls_embedding']
        
        result = {
            'cls_embedding': cls_embedding,
            'sequence_embeddings': encoder_output['sequence_embeddings'],
        }
        
        # Classification
        if self.use_classification:
            result['class_logits'] = self.classification_head(cls_embedding)
        
        # Regression
        if self.use_regression:
            result['concentrations'] = self.regression_head(cls_embedding)
        
        return result


if __name__ == "__main__":
    batch_size = 4
    d_model = 256
    n_classes = 5
    n_bins = 2048
    
    # Test ClassificationHead
    print("Testing ClassificationHead...")
    cls_head = ClassificationHead(d_model, n_classes)
    cls_input = torch.randn(batch_size, d_model)
    cls_output = cls_head(cls_input)
    print(f"  Input shape: {cls_input.shape}")
    print(f"  Output shape: {cls_output.shape}")
    
    # Test RegressionHead
    print("\nTesting RegressionHead...")
    reg_head = RegressionHead(d_model, n_classes)
    reg_output = reg_head(cls_input)
    print(f"  Output shape: {reg_output.shape}")
    print(f"  Output range: [{reg_output.min():.4f}, {reg_output.max():.4f}]")
    
    # Test MaskedPredictionHead
    print("\nTesting MaskedPredictionHead...")
    mip_head = MaskedPredictionHead(d_model)
    seq_input = torch.randn(batch_size, n_bins, d_model)
    mip_output = mip_head(seq_input)
    print(f"  Input shape: {seq_input.shape}")
    print(f"  Output shape: {mip_output.shape}")
    
    # Test MultiTaskHead
    print("\nTesting MultiTaskHead...")
    multi_head = MultiTaskHead(d_model, n_classes, n_classes)
    multi_output = multi_head(cls_input)
    print(f"  Class logits shape: {multi_output['class_logits'].shape}")
    print(f"  Concentrations shape: {multi_output['concentrations'].shape}")
