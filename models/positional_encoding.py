"""
Positional Encoding for LIBS Transformer.

Implements sinusoidal positional encoding for wavelength positions.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as introduced in "Attention is All You Need".
    
    Encodes position information using sine and cosine functions at different frequencies.
    This allows the model to learn relative positions between wavelength bins.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length (number of wavelength bins + 1 for CLS)
        dropout: Dropout rate applied to the encoding
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2049,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Uses learned embeddings for each position instead of fixed sinusoidal patterns.
    Can be more flexible but requires more parameters.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2049,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WavelengthEncoding(nn.Module):
    """
    Wavelength-aware positional encoding.
    
    Instead of integer positions, uses actual wavelength values 
    to create more physically meaningful encodings.
    
    Args:
        d_model: Model dimension
        wavelength_min: Minimum wavelength value
        wavelength_max: Maximum wavelength value
        n_bins: Number of wavelength bins
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        wavelength_min: float = 200.0,
        wavelength_max: float = 900.0,
        n_bins: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create wavelength values for each bin
        wavelengths = torch.linspace(wavelength_min, wavelength_max, n_bins)
        
        # Normalize to [0, 1] for encoding
        wavelengths_normalized = (wavelengths - wavelength_min) / (wavelength_max - wavelength_min)
        
        # Create encoding similar to sinusoidal but using wavelength values
        pe = torch.zeros(n_bins + 1, d_model)  # +1 for CLS token
        
        # CLS token gets position 0 encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Position 0 for CLS
        pe[0, 0::2] = torch.sin(torch.zeros(1) * div_term)
        pe[0, 1::2] = torch.cos(torch.zeros(1) * div_term)
        
        # Wavelength-based encoding for spectral bins
        for i, wl in enumerate(wavelengths_normalized):
            pe[i + 1, 0::2] = torch.sin(wl * 1000 * div_term)
            pe[i + 1, 1::2] = torch.cos(wl * 1000 * div_term)
        
        pe = pe.unsqueeze(0)  # [1, n_bins+1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add wavelength-based positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test positional encodings
    batch_size = 4
    seq_len = 2049  # 2048 bins + 1 CLS token
    d_model = 256
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("Testing SinusoidalPositionalEncoding...")
    sinusoidal = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
    out = sinusoidal(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    print("\nTesting LearnablePositionalEncoding...")
    learnable = LearnablePositionalEncoding(d_model, max_len=seq_len)
    out = learnable(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in learnable.parameters())}")
    
    print("\nTesting WavelengthEncoding...")
    wavelength = WavelengthEncoding(d_model, n_bins=2048)
    out = wavelength(x)
    print(f"  Output shape: {out.shape}")
