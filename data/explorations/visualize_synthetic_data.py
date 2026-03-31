"""
Visualization script for synthetic LIBS data.

Run this script to generate exploratory plots of the synthetic LIBS spectra.
Outputs are saved to data/explorations/figures/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data.synthetic_generator import SyntheticLIBSGenerator, MATERIAL_CLASSES


def setup_output_dir():
    """Create output directory for figures."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_all_classes(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Plot example spectra for all 5 material classes."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Synthetic LIBS Spectra - All Material Classes", fontsize=14, fontweight='bold')
    
    wavelengths = np.arange(generator.n_bins)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for class_id, ax in enumerate(axes):
        material = MATERIAL_CLASSES[class_id]
        spectrum, _, _ = generator.generate_spectrum(class_id=class_id)
        
        ax.fill_between(wavelengths, spectrum, alpha=0.3, color=colors[class_id])
        ax.plot(wavelengths, spectrum, color=colors[class_id], linewidth=0.8)
        
        # Mark peak positions
        for peak in material.peaks:
            ax.axvline(peak.position, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        
        ax.set_ylabel("Intensity", fontsize=10)
        ax.set_title(f"Class {class_id}: {material.name}", fontsize=11, loc='left')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Wavelength Bin", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "01_all_classes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_all_classes.png")


def plot_class_comparison(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Plot all classes overlaid for comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    wavelengths = np.arange(generator.n_bins)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for class_id in range(5):
        material = MATERIAL_CLASSES[class_id]
        # Generate without noise for cleaner comparison
        spectrum, _, _ = generator.generate_spectrum(
            class_id=class_id, 
            add_noise=False,
            add_baseline=False
        )
        ax.plot(wavelengths, spectrum, color=colors[class_id], 
                linewidth=1.5, label=f"{material.name}", alpha=0.8)
    
    ax.set_xlabel("Wavelength Bin", fontsize=11)
    ax.set_ylabel("Intensity", fontsize=11)
    ax.set_title("Class Comparison (No Noise, No Baseline)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, generator.n_bins)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_class_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_class_comparison.png")


def plot_noise_effect(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Show effect of noise on spectra."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Effect of Noise and Baseline", fontsize=14, fontweight='bold')
    
    wavelengths = np.arange(generator.n_bins)
    class_id = 0  # Use Iron for demo
    
    configs = [
        ("Clean (No Noise, No Baseline)", {"add_noise": False, "add_baseline": False}),
        ("With Baseline Only", {"add_noise": False, "add_baseline": True}),
        ("With Noise Only", {"add_noise": True, "add_baseline": False}),
        ("Full (Noise + Baseline)", {"add_noise": True, "add_baseline": True}),
    ]
    
    # Use same seed for fair comparison
    for ax, (title, kwargs) in zip(axes.flat, configs):
        gen = SyntheticLIBSGenerator(seed=42)
        spectrum, _, _ = gen.generate_spectrum(class_id=class_id, **kwargs)
        
        ax.fill_between(wavelengths, spectrum, alpha=0.3, color='steelblue')
        ax.plot(wavelengths, spectrum, color='steelblue', linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Wavelength Bin")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_noise_effect.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_noise_effect.png")


def plot_intra_class_variation(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Show variation within the same class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Intra-Class Variation (Same Class, Different Samples)", fontsize=14, fontweight='bold')
    
    wavelengths = np.arange(generator.n_bins)
    
    for row, class_id in enumerate([0, 2]):  # Iron and Aluminum
        material = MATERIAL_CLASSES[class_id]
        for col in range(3):
            ax = axes[row, col]
            spectrum, _, _ = generator.generate_spectrum(class_id=class_id)
            
            ax.fill_between(wavelengths, spectrum, alpha=0.3, color=f'C{class_id}')
            ax.plot(wavelengths, spectrum, color=f'C{class_id}', linewidth=0.8)
            ax.set_title(f"{material.name} - Sample {col+1}", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            if row == 1:
                ax.set_xlabel("Wavelength Bin")
            if col == 0:
                ax.set_ylabel("Intensity")
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_intra_class_variation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_intra_class_variation.png")


def plot_mixed_spectra(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Plot examples of mixed spectra with different compositions."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    fig.suptitle("Mixed Spectra Examples", fontsize=14, fontweight='bold')
    
    wavelengths = np.arange(generator.n_bins)
    
    # Define specific mixtures
    mixtures = [
        ("50% Iron + 50% Copper", np.array([0.5, 0.5, 0.0, 0.0, 0.0])),
        ("70% Aluminum + 30% Calcium", np.array([0.0, 0.0, 0.7, 0.3, 0.0])),
        ("Equal mix (20% each)", np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        ("40% Fe + 30% Cu + 30% Al", np.array([0.4, 0.3, 0.3, 0.0, 0.0])),
    ]
    
    for idx, (title, concentrations) in enumerate(mixtures):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        spectrum, _, _ = generator.generate_spectrum(concentrations=concentrations)
        
        ax.fill_between(wavelengths, spectrum, alpha=0.3, color='purple')
        ax.plot(wavelengths, spectrum, color='purple', linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Wavelength Bin")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add concentration bar
        ax_inset = ax.inset_axes([0.02, 0.7, 0.15, 0.25])
        ax_inset.barh(range(5), concentrations, color=plt.cm.tab10(np.arange(5)/10))
        ax_inset.set_yticks(range(5))
        ax_inset.set_yticklabels(['Fe', 'Cu', 'Al', 'Ca', 'Mix'], fontsize=7)
        ax_inset.set_xlim(0, 1)
        ax_inset.set_xticks([0, 0.5, 1])
        ax_inset.tick_params(axis='x', labelsize=6)
        ax_inset.set_title("Conc.", fontsize=7)
    
    # Random mixed spectra
    ax = fig.add_subplot(gs[2, :])
    for i in range(5):
        spectrum, dominant, conc = generator.generate_mixed_spectrum(n_components=3)
        ax.plot(wavelengths, spectrum + i*0.3, alpha=0.7, linewidth=0.8,
                label=f"Mix {i+1} (dominant: {MATERIAL_CLASSES[dominant].name})")
    
    ax.set_xlabel("Wavelength Bin")
    ax.set_ylabel("Intensity (stacked)")
    ax.set_title("Random 3-Component Mixtures (Stacked)", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "05_mixed_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_mixed_spectra.png")


def plot_peak_positions(output_dir: Path):
    """Visualize peak positions for all classes."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    y_positions = []
    
    for class_id, material in MATERIAL_CLASSES.items():
        y = class_id
        y_positions.append(y)
        
        for peak in material.peaks:
            # Size proportional to intensity
            size = peak.base_intensity * 300
            ax.scatter(peak.position, y, s=size, c=[colors[class_id]], 
                      alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{MATERIAL_CLASSES[i].name}" for i in range(5)])
    ax.set_xlabel("Wavelength Bin (Peak Position)", fontsize=11)
    ax.set_title("Peak Positions by Material Class (Size = Intensity)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2048)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for intensities
    for intensity, label in [(0.4, "Low"), (0.7, "Medium"), (0.95, "High")]:
        ax.scatter([], [], s=intensity*300, c='gray', alpha=0.7, 
                  edgecolors='black', label=f"{label} ({intensity})")
    ax.legend(title="Intensity", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_peak_positions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_peak_positions.png")


def plot_dataset_statistics(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Generate and visualize dataset statistics."""
    # Generate a sample dataset
    print("Generating sample dataset for statistics...")
    spectra, labels, concentrations = generator.generate_dataset(
        n_samples=1000,
        mixed_ratio=0.2,
    )
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle("Dataset Statistics (n=1000, 20% mixed)", fontsize=14, fontweight='bold')
    
    # Class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    class_names = [MATERIAL_CLASSES[i].name for i in range(5)]
    counts = np.bincount(labels, minlength=5)
    bars = ax1.bar(class_names, counts, color=plt.cm.tab10(np.arange(5)/10))
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution")
    ax1.bar_label(bars)
    
    # Mean spectrum per class
    ax2 = fig.add_subplot(gs[0, 1:])
    wavelengths = np.arange(generator.n_bins)
    for class_id in range(5):
        mask = labels == class_id
        mean_spectrum = spectra[mask].mean(axis=0)
        ax2.plot(wavelengths, mean_spectrum, label=MATERIAL_CLASSES[class_id].name, 
                linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Wavelength Bin")
    ax2.set_ylabel("Mean Intensity")
    ax2.set_title("Mean Spectrum per Class")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Intensity histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(spectra.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Intensity Distribution")
    ax3.axvline(spectra.mean(), color='red', linestyle='--', label=f'Mean: {spectra.mean():.3f}')
    ax3.legend()
    
    # Concentration distribution (for mixed samples)
    ax4 = fig.add_subplot(gs[1, 1])
    mixed_mask = concentrations.max(axis=1) < 0.99  # Mixed samples
    if mixed_mask.sum() > 0:
        mixed_conc = concentrations[mixed_mask]
        parts = ax4.violinplot([mixed_conc[:, i] for i in range(5)], 
                               positions=range(5), showmeans=True)
        ax4.set_xticks(range(5))
        ax4.set_xticklabels([m.name[:3] for m in MATERIAL_CLASSES.values()])
        ax4.set_ylabel("Concentration")
        ax4.set_title(f"Concentration Distribution\n(Mixed Samples, n={mixed_mask.sum()})")
    
    # Sample spectra grid
    ax5 = fig.add_subplot(gs[1, 2])
    # Show 10 random spectra stacked
    sample_idx = np.random.choice(len(spectra), 10, replace=False)
    for i, idx in enumerate(sample_idx):
        ax5.plot(wavelengths, spectra[idx] + i*0.15, alpha=0.7, linewidth=0.5)
    ax5.set_xlabel("Wavelength Bin")
    ax5.set_ylabel("Intensity (stacked)")
    ax5.set_title("Random Sample Spectra")
    ax5.set_yticks([])
    
    plt.savefig(output_dir / "07_dataset_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_dataset_statistics.png")


def plot_masking_preview(generator: SyntheticLIBSGenerator, output_dir: Path):
    """Preview peak-biased contiguous masking for pre-training."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Peak-Biased Contiguous Masking (Pre-training Objective)", fontsize=14, fontweight='bold')
    
    wavelengths = np.arange(generator.n_bins)
    spectrum, _, _ = generator.generate_spectrum(class_id=0)
    
    # Parameters
    mask_ratio = 0.15
    block_sizes = [25, 50, 100]
    peak_threshold = 0.2
    peak_bias_ratio = 0.5
    n_bins = len(spectrum)
    n_masked_total = int(n_bins * mask_ratio)
    
    rng = np.random.default_rng(42)
    
    # Detect peaks
    peak_mask = spectrum > peak_threshold
    # Expand peak regions
    expanded_peaks = np.zeros_like(peak_mask)
    for i in range(len(peak_mask)):
        if peak_mask[i]:
            start = max(0, i - 5)
            end = min(len(peak_mask), i + 6)
            expanded_peaks[start:end] = True
    
    # Original spectrum with peak regions highlighted
    axes[0].fill_between(wavelengths, spectrum, alpha=0.3, color='steelblue')
    axes[0].plot(wavelengths, spectrum, color='steelblue', linewidth=0.8)
    # Highlight peak regions
    peak_indices = np.where(expanded_peaks)[0]
    if len(peak_indices) > 0:
        axes[0].fill_between(wavelengths, 0, 0.1, where=expanded_peaks, 
                            color='orange', alpha=0.5, label='Peak regions')
    axes[0].axhline(peak_threshold, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Peak threshold ({peak_threshold})')
    axes[0].set_title("Original Spectrum (orange = detected peak regions)", loc='left')
    axes[0].set_ylabel("Intensity")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='upper right', fontsize=8)
    
    # Create peak-biased mask
    mask = np.zeros(n_bins, dtype=bool)
    region_info = []  # (start, end, on_peak)
    
    # First, place some blocks on peaks
    n_peak_masked = int(n_masked_total * peak_bias_ratio)
    peak_positions = np.where(expanded_peaks)[0]
    
    masked_count = 0
    attempts = 0
    while masked_count < n_peak_masked and attempts < 50 and len(peak_positions) > 0:
        block_size = rng.choice(block_sizes)
        # Find valid starts within peak regions
        valid_starts = peak_positions[peak_positions <= n_bins - block_size]
        if len(valid_starts) == 0:
            attempts += 1
            continue
        start = rng.choice(valid_starts)
        end = start + block_size
        if not mask[start:end].any():
            mask[start:end] = True
            masked_count += block_size
            region_info.append((start, end, True))
        attempts += 1
    
    # Then place remaining blocks anywhere
    remaining = n_masked_total - mask.sum()
    attempts = 0
    while remaining > 0 and attempts < 50:
        block_size = rng.choice(block_sizes)
        start = rng.integers(0, n_bins - block_size)
        end = start + block_size
        if not mask[start:end].any():
            mask[start:end] = True
            remaining -= block_size
            on_peak = expanded_peaks[start:end].any()
            region_info.append((start, end, on_peak))
        attempts += 1
    
    # Masked spectrum (model input)
    masked_spectrum = spectrum.copy()
    masked_spectrum[mask] = 0
    
    axes[1].fill_between(wavelengths, masked_spectrum, alpha=0.3, color='steelblue')
    axes[1].plot(wavelengths, masked_spectrum, color='steelblue', linewidth=0.8)
    # Highlight masked regions (different colors for peak vs non-peak)
    for start, end, on_peak in region_info:
        color = 'green' if on_peak else 'red'
        axes[1].axvspan(start, end, color=color, alpha=0.3)
    axes[1].set_title(f"Masked Spectrum - Variable blocks {block_sizes} (green=on peak, red=off peak)", loc='left')
    axes[1].set_ylabel("Intensity")
    axes[1].set_ylim(0, 1.1)
    
    # Show masked intensities (what needs to be predicted)
    peak_masked = mask & expanded_peaks
    nonpeak_masked = mask & ~expanded_peaks
    
    axes[2].bar(wavelengths[peak_masked], spectrum[peak_masked], width=1, 
               color='green', alpha=0.7, label='Peak bins')
    axes[2].bar(wavelengths[nonpeak_masked], spectrum[nonpeak_masked], width=1, 
               color='red', alpha=0.7, label='Non-peak bins')
    axes[2].set_title("Target: Masked Intensities to Predict", loc='left')
    axes[2].set_ylabel("Intensity")
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc='upper right', fontsize=8)
    
    # Statistics
    peak_coverage = peak_masked.sum() / max(mask.sum(), 1)
    n_regions = len(region_info)
    n_peak_regions = sum(1 for _, _, on_peak in region_info if on_peak)
    
    axes[3].axis('off')
    stats_text = f"""
    Masking Statistics:
    ─────────────────────────────
    Total masked bins:     {mask.sum()} / {n_bins} ({mask.sum()/n_bins:.1%})
    Number of regions:     {n_regions}
    Block sizes used:      {block_sizes}
    
    Peak Coverage:
    ─────────────────────────────
    Bins on peaks:         {peak_masked.sum()} ({peak_coverage:.1%} of masked)
    Bins off peaks:        {nonpeak_masked.sum()} ({1-peak_coverage:.1%} of masked)
    Regions on peaks:      {n_peak_regions} / {n_regions}
    Target peak ratio:     {peak_bias_ratio:.0%}
    
    This ensures the model learns to predict
    actual spectral features, not just noise!
    """
    axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    for ax in axes[:3]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "08_masking_preview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_masking_preview.png")


def main():
    """Run all visualizations."""
    print("=" * 60)
    print("LIBS Synthetic Data Visualization")
    print("=" * 60)
    
    # Setup
    output_dir = setup_output_dir()
    print(f"\nOutput directory: {output_dir}")
    
    generator = SyntheticLIBSGenerator(seed=42)
    
    print("\nMaterial Classes:")
    for cid, info in generator.get_class_info().items():
        print(f"  {cid}: {info['name']:10s} - {info['n_peaks']} peaks at {info['peak_positions']}")
    
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    print("-" * 60)
    
    # Generate all plots
    plot_all_classes(generator, output_dir)
    plot_class_comparison(generator, output_dir)
    plot_noise_effect(generator, output_dir)
    plot_intra_class_variation(generator, output_dir)
    plot_mixed_spectra(generator, output_dir)
    plot_peak_positions(output_dir)
    plot_dataset_statistics(generator, output_dir)
    plot_masking_preview(generator, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
