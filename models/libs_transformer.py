"""
LIBS Transformer Model.

Main transformer encoder architecture for LIBS spectral analysis.
Adapted for continuous intensity values with masked intensity prediction.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .positional_encoding import SinusoidalPositionalEncoding


class SpectralEmbedding(nn.Module):
    """
    Embedding layer for LIBS spectra.
    
    Projects each intensity value to the model dimension and adds
    positional encoding for wavelength positions.
    
    Args:
        d_model: Model dimension
        n_bins: Number of wavelength bins
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_bins: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bins = n_bins
        
        # Project each intensity value (scalar) to d_model dimensions
        self.intensity_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Learnable CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Learnable MASK token for masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=n_bins + 1,  # +1 for CLS token
            dropout=dropout,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed spectrum and add positional encoding.
        
        Args:
            x: Input spectrum of shape [batch_size, n_bins]
            mask: Optional boolean mask of shape [batch_size, n_bins]
                  indicating positions to mask (True = masked)
        
        Returns:
            Embedded sequence of shape [batch_size, n_bins + 1, d_model]
        """
        batch_size = x.shape[0]
        
        # Expand intensity to [batch, n_bins, 1]
        x = x.unsqueeze(-1)
        
        # Project intensities: [batch, n_bins, d_model]
        x = self.intensity_projection(x)
        
        # Apply mask token to masked positions if mask is provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            mask_token_expanded = self.mask_token.expand(batch_size, self.n_bins, -1)
            x = torch.where(mask_expanded, mask_token_expanded, x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, n_bins + 1, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.
    
    Consists of:
    1. Multi-head self-attention with pre-LayerNorm
    2. Feed-forward network with pre-LayerNorm
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            attn_mask: Optional attention mask
            key_padding_mask: Optional padding mask
            need_weights: Whether to return attention weights (uses more memory)
            
        Returns:
            Tuple of (output, attention_weights)
            - output: Shape [batch_size, seq_len, d_model]
            - attention_weights: Shape [batch_size, n_heads, seq_len, seq_len] or None
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + ff_output
        
        return x, attn_weights


class LIBSTransformer(nn.Module):
    """
    LIBS Foundation Model Transformer.
    
    A transformer encoder for learning representations from LIBS spectra.
    Supports masked intensity prediction for self-supervised pre-training.
    
    Args:
        n_bins: Number of wavelength bins
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        n_classes: Number of classes for downstream classification
    """
    
    def __init__(
        self,
        n_bins: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 5,
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Spectral embedding
        self.embedding = SpectralEmbedding(
            d_model=d_model,
            n_bins=n_bins,
            dropout=dropout,
        )
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Masked intensity prediction head
        self.mip_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input spectrum of shape [batch_size, n_bins]
            mask: Optional boolean mask of shape [batch_size, n_bins]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with:
            - 'cls_embedding': CLS token embedding [batch_size, d_model]
            - 'sequence_output': Full sequence output [batch_size, n_bins + 1, d_model]
            - 'mip_predictions': Masked intensity predictions [batch_size, n_bins]
            - 'attention_weights': (optional) List of attention weights per layer
        """
        # Embed input
        hidden = self.embedding(x, mask=mask)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            hidden, attn = block(hidden, need_weights=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # Final normalization
        hidden = self.final_norm(hidden)
        
        # Extract CLS embedding (first token)
        cls_embedding = hidden[:, 0, :]
        
        # Extract sequence embeddings (excluding CLS)
        sequence_embeddings = hidden[:, 1:, :]
        
        # Masked intensity prediction
        mip_predictions = self.mip_head(sequence_embeddings).squeeze(-1)
        
        result = {
            'cls_embedding': cls_embedding,
            'sequence_output': hidden,
            'sequence_embeddings': sequence_embeddings,
            'mip_predictions': mip_predictions,
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get CLS embeddings for downstream tasks.
        
        Args:
            x: Input spectrum of shape [batch_size, n_bins]
            
        Returns:
            CLS embeddings of shape [batch_size, d_model]
        """
        output = self.forward(x, mask=None, return_attention=False)
        return output['cls_embedding']
    
    def compute_mip_loss(
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
            mask: Boolean mask indicating masked positions [batch_size, n_bins]
            
        Returns:
            Mean squared error loss over masked positions
        """
        # Only compute loss on masked positions
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        
        if masked_predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        loss = nn.functional.mse_loss(masked_predictions, masked_targets)
        return loss
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.final_norm.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_bins = 2048
    d_model = 256
    
    model = LIBSTransformer(
        n_bins=n_bins,
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    )
    
    print(f"Model Parameters: {model.num_parameters:,}")
    
    # Test forward pass without mask
    x = torch.rand(batch_size, n_bins)
    output = model(x)
    print(f"\nForward pass (no mask):")
    print(f"  CLS embedding shape: {output['cls_embedding'].shape}")
    print(f"  Sequence output shape: {output['sequence_output'].shape}")
    print(f"  MIP predictions shape: {output['mip_predictions'].shape}")
    
    # Test forward pass with mask
    mask = torch.rand(batch_size, n_bins) < 0.15
    output = model(x, mask=mask, return_attention=True)
    print(f"\nForward pass (with mask):")
    print(f"  CLS embedding shape: {output['cls_embedding'].shape}")
    print(f"  MIP predictions shape: {output['mip_predictions'].shape}")
    print(f"  Number of attention layers: {len(output['attention_weights'])}")
    print(f"  Attention shape per layer: {output['attention_weights'][0].shape}")
    
    # Test MIP loss computation
    targets = torch.rand(batch_size, n_bins)
    loss = model.compute_mip_loss(output['mip_predictions'], targets, mask)
    print(f"\nMIP Loss: {loss.item():.4f}")
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(x)
    print(f"\nEmbeddings shape: {embeddings.shape}")
