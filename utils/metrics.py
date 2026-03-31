"""
Evaluation metrics and visualization utilities for LIBS Foundation Model.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_classification_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class labels or logits [n_samples] or [n_samples, n_classes]
        targets: True class labels [n_samples]
        n_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert logits to predictions if needed
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'balanced_accuracy': balanced_accuracy_score(targets, predictions),
        'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(targets, predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == targets[mask]).mean()
            class_name = class_names[c] if class_names else f'class_{c}'
            metrics[f'accuracy_{class_name}'] = class_acc
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm
    
    return metrics


def compute_regression_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_outputs: int = 5,
    output_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        predictions: Predicted values [n_samples, n_outputs]
        targets: True values [n_samples, n_outputs]
        n_outputs: Number of output dimensions
        output_names: Optional list of output names
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Ensure 2D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    # Overall metrics
    metrics = {
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions),
    }
    
    # Per-output metrics
    for i in range(min(n_outputs, predictions.shape[1])):
        output_name = output_names[i] if output_names else f'output_{i}'
        metrics[f'mse_{output_name}'] = mean_squared_error(targets[:, i], predictions[:, i])
        metrics[f'mae_{output_name}'] = mean_absolute_error(targets[:, i], predictions[:, i])
        metrics[f'r2_{output_name}'] = r2_score(targets[:, i], predictions[:, i])
    
    return metrics


def compute_mip_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute metrics for masked intensity prediction.
    
    Args:
        predictions: Predicted intensities [batch_size, n_bins]
        targets: True intensities [batch_size, n_bins]
        mask: Boolean mask indicating masked positions [batch_size, n_bins]
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Select only masked positions
    masked_preds = predictions[mask]
    masked_targets = targets[mask]
    
    if len(masked_preds) == 0:
        return {'mse': 0.0, 'mae': 0.0, 'r2': 0.0}
    
    return {
        'mse': mean_squared_error(masked_targets, masked_preds),
        'rmse': np.sqrt(mean_squared_error(masked_targets, masked_preds)),
        'mae': mean_absolute_error(masked_targets, masked_preds),
        'r2': r2_score(masked_targets, masked_preds) if len(masked_targets) > 1 else 0.0,
    }


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for each line
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            continue  # Skip confusion matrix in string format
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return '\n'.join(lines)


# ============================================================================
# Visualization utilities
# ============================================================================

def plot_spectrum(
    spectrum: np.ndarray,
    title: str = 'LIBS Spectrum',
    wavelength_range: Optional[Tuple[float, float]] = None,
    ax=None,
    **kwargs,
):
    """
    Plot a single spectrum.
    
    Args:
        spectrum: Spectrum array [n_bins]
        title: Plot title
        wavelength_range: Optional (min, max) wavelength range
        ax: Optional matplotlib axes
        **kwargs: Additional kwargs for plot
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    n_bins = len(spectrum)
    
    if wavelength_range:
        x = np.linspace(wavelength_range[0], wavelength_range[1], n_bins)
        ax.set_xlabel('Wavelength (nm)')
    else:
        x = np.arange(n_bins)
        ax.set_xlabel('Bin Index')
    
    ax.plot(x, spectrum, **kwargs)
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    
    return ax


def plot_spectra_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = 'Spectrum Reconstruction',
    ax=None,
):
    """
    Plot comparison between original and reconstructed spectrum.
    
    Args:
        original: Original spectrum [n_bins]
        reconstructed: Reconstructed spectrum [n_bins]
        mask: Optional boolean mask highlighting regions [n_bins]
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    n_bins = len(original)
    x = np.arange(n_bins)
    
    ax.plot(x, original, label='Original', alpha=0.8)
    ax.plot(x, reconstructed, label='Reconstructed', alpha=0.8)
    
    if mask is not None:
        # Highlight masked regions
        masked_regions = np.where(mask)[0]
        if len(masked_regions) > 0:
            ax.scatter(masked_regions, original[masked_regions], 
                      c='red', s=10, alpha=0.5, label='Masked', zorder=5)
    
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    ax=None,
    cmap: str = 'Blues',
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix [n_classes, n_classes]
        class_names: Optional class names
        normalize: Whether to normalize by row (true class)
        title: Plot title
        ax: Optional matplotlib axes
        cmap: Colormap name
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
    else:
        cm_normalized = cm
        fmt = 'd'
    
    n_classes = len(cm)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    return ax


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    title: str = 't-SNE of Embeddings',
    perplexity: int = 30,
    ax=None,
):
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding vectors [n_samples, d_model]
        labels: Optional class labels [n_samples]
        class_names: Optional class names
        title: Plot title
        perplexity: t-SNE perplexity parameter
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names[label] if class_names else f'Class {label}'
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=name,
                alpha=0.7,
            )
        ax.legend()
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    
    return ax


def plot_regression_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_idx: int = 0,
    output_name: Optional[str] = None,
    title: Optional[str] = None,
    ax=None,
):
    """
    Plot scatter plot of predictions vs targets for regression.
    
    Args:
        predictions: Predicted values [n_samples, n_outputs]
        targets: True values [n_samples, n_outputs]
        output_idx: Which output to plot
        output_name: Name of the output
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    pred = predictions[:, output_idx] if predictions.ndim > 1 else predictions
    targ = targets[:, output_idx] if targets.ndim > 1 else targets
    
    ax.scatter(targ, pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(targ.min(), pred.min())
    max_val = max(targ.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    
    # Compute R²
    r2 = r2_score(targ, pred)
    
    name = output_name or f'Output {output_idx}'
    if title is None:
        title = f'{name}: R² = {r2:.4f}'
    
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = 'Training Curves',
    ax=None,
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        title: Plot title
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


if __name__ == "__main__":
    # Test metrics
    print("Testing classification metrics...")
    preds = np.random.randint(0, 5, 100)
    targets = np.random.randint(0, 5, 100)
    cls_metrics = compute_classification_metrics(preds, targets)
    print(format_metrics(cls_metrics))
    
    print("\nTesting regression metrics...")
    preds = np.random.rand(100, 5)
    targets = np.random.rand(100, 5)
    reg_metrics = compute_regression_metrics(preds, targets)
    print(format_metrics(reg_metrics))
    
    print("\nTesting MIP metrics...")
    preds = np.random.rand(10, 2048)
    targets = np.random.rand(10, 2048)
    mask = np.random.rand(10, 2048) < 0.15
    mip_metrics = compute_mip_metrics(preds, targets, mask)
    print(format_metrics(mip_metrics))
