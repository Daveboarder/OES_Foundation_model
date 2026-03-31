"""
Task-specific heads for LIBS Foundation Model.

Includes heads for:
- Classification (material identification)
- Regression (concentration prediction)
- Masked intensity prediction
"""

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
