"""
Synthetic LIBS Data Generator

Generates realistic LIBS-like spectra with 5 material classes,
each with characteristic emission peaks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class PeakDefinition:
    """Definition of a spectral peak."""
    position: int          # Bin index (0-2047)
    base_intensity: float  # Base intensity (0-1)
    width_range: Tuple[float, float] = (5.0, 15.0)  # Width range in bins


@dataclass
class MaterialClass:
    """Definition of a material class with its characteristic peaks."""
    name: str
    peaks: List[PeakDefinition]
    description: str = ""


# Define the 5 material classes with characteristic peaks
MATERIAL_CLASSES: Dict[int, MaterialClass] = {
    0: MaterialClass(
        name="Iron",
        description="Iron-dominant sample with Fe emission lines",
        peaks=[
            PeakDefinition(position=300, base_intensity=0.9),
            PeakDefinition(position=320, base_intensity=0.6),
            PeakDefinition(position=520, base_intensity=0.85),
            PeakDefinition(position=545, base_intensity=0.5),
            PeakDefinition(position=850, base_intensity=0.75),
            PeakDefinition(position=880, base_intensity=0.4),
        ]
    ),
    1: MaterialClass(
        name="Copper",
        description="Copper-dominant sample with Cu emission lines",
        peaks=[
            PeakDefinition(position=200, base_intensity=0.85),
            PeakDefinition(position=230, base_intensity=0.55),
            PeakDefinition(position=680, base_intensity=0.9),
            PeakDefinition(position=720, base_intensity=0.6),
            PeakDefinition(position=1200, base_intensity=0.8),
            PeakDefinition(position=1250, base_intensity=0.45),
        ]
    ),
    2: MaterialClass(
        name="Aluminum",
        description="Aluminum-dominant sample with Al emission lines",
        peaks=[
            PeakDefinition(position=400, base_intensity=0.95),
            PeakDefinition(position=430, base_intensity=0.5),
            PeakDefinition(position=900, base_intensity=0.8),
            PeakDefinition(position=940, base_intensity=0.55),
            PeakDefinition(position=1500, base_intensity=0.85),
            PeakDefinition(position=1540, base_intensity=0.4),
        ]
    ),
    3: MaterialClass(
        name="Calcium",
        description="Calcium-dominant sample with Ca emission lines",
        peaks=[
            PeakDefinition(position=600, base_intensity=0.9),
            PeakDefinition(position=635, base_intensity=0.65),
            PeakDefinition(position=1100, base_intensity=0.85),
            PeakDefinition(position=1140, base_intensity=0.5),
            PeakDefinition(position=1700, base_intensity=0.8),
            PeakDefinition(position=1750, base_intensity=0.45),
        ]
    ),
    4: MaterialClass(
        name="Mixed",
        description="Complex mixed sample with multiple elements",
        peaks=[
            # Mix of peaks from multiple elements
            PeakDefinition(position=300, base_intensity=0.5),   # Fe
            PeakDefinition(position=680, base_intensity=0.55),  # Cu
            PeakDefinition(position=900, base_intensity=0.5),   # Al
            PeakDefinition(position=1100, base_intensity=0.6),  # Ca
            PeakDefinition(position=1800, base_intensity=0.45), # Unique to mixed
            PeakDefinition(position=1900, base_intensity=0.4),  # Unique to mixed
        ]
    ),
}


class SyntheticLIBSGenerator:
    """
    Generator for synthetic LIBS spectra.
    
    Creates realistic LIBS-like spectra with:
    - Characteristic emission peaks per material class
    - Variable peak widths and intensities
    - Baseline continuum
    - Shot noise and Gaussian noise
    - Optional mixing of multiple materials
    """
    
    def __init__(
        self,
        n_bins: int = 2048,
        noise_sigma: float = 0.02,
        peak_width_range: Tuple[float, float] = (5.0, 15.0),
        intensity_variation: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic LIBS generator.
        
        Args:
            n_bins: Number of wavelength bins
            noise_sigma: Standard deviation of Gaussian noise
            peak_width_range: Range of peak widths (min, max) in bins
            intensity_variation: Fractional variation in peak intensities (+/-)
            seed: Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.noise_sigma = noise_sigma
        self.peak_width_range = peak_width_range
        self.intensity_variation = intensity_variation
        self.rng = np.random.default_rng(seed)
        
        # Precompute bin indices for efficiency
        self.bin_indices = np.arange(n_bins)
    
    def _gaussian_peak(
        self, 
        position: float, 
        width: float, 
        intensity: float
    ) -> np.ndarray:
        """Generate a Gaussian peak profile."""
        return intensity * np.exp(-0.5 * ((self.bin_indices - position) / width) ** 2)
    
    def _generate_baseline(self) -> np.ndarray:
        """Generate a smooth baseline continuum using polynomial."""
        # Random polynomial coefficients for baseline
        x = self.bin_indices / self.n_bins  # Normalize to [0, 1]
        
        # Random coefficients for cubic polynomial
        a = self.rng.uniform(0.02, 0.08)
        b = self.rng.uniform(-0.05, 0.05)
        c = self.rng.uniform(-0.02, 0.02)
        d = self.rng.uniform(0.01, 0.05)
        
        baseline = a + b * x + c * x**2 + d * x**3
        return np.clip(baseline, 0, None)
    
    def _add_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Add shot noise (Poisson-like) and Gaussian noise."""
        # Shot noise (proportional to sqrt of intensity)
        shot_noise = self.rng.normal(0, np.sqrt(np.abs(spectrum) + 1e-6) * 0.05)
        
        # Gaussian noise
        gaussian_noise = self.rng.normal(0, self.noise_sigma, self.n_bins)
        
        return spectrum + shot_noise + gaussian_noise
    
    def generate_single_class_spectrum(
        self,
        class_id: int,
        concentration: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a spectrum for a single material class.
        
        Args:
            class_id: Material class ID (0-4)
            concentration: Concentration scaling factor (0-1)
            
        Returns:
            Spectrum array of shape (n_bins,)
        """
        if class_id not in MATERIAL_CLASSES:
            raise ValueError(f"Invalid class_id: {class_id}. Must be 0-4.")
        
        material = MATERIAL_CLASSES[class_id]
        spectrum = np.zeros(self.n_bins)
        
        # Add characteristic peaks
        for peak in material.peaks:
            # Vary peak width
            width = self.rng.uniform(*self.peak_width_range)
            
            # Vary peak intensity
            intensity_factor = 1.0 + self.rng.uniform(
                -self.intensity_variation, 
                self.intensity_variation
            )
            intensity = peak.base_intensity * intensity_factor * concentration
            
            # Slight position variation (1-2 bins)
            position = peak.position + self.rng.uniform(-2, 2)
            
            spectrum += self._gaussian_peak(position, width, intensity)
        
        return spectrum
    
    def generate_spectrum(
        self,
        class_id: Optional[int] = None,
        concentrations: Optional[np.ndarray] = None,
        add_baseline: bool = True,
        add_noise: bool = True,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Generate a synthetic LIBS spectrum.
        
        Args:
            class_id: If provided, generate spectrum for this class.
                     If None and concentrations is None, randomly select a class.
            concentrations: Array of shape (n_classes,) with concentration weights.
                           If provided, creates a mixed spectrum.
            add_baseline: Whether to add baseline continuum
            add_noise: Whether to add noise
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Tuple of (spectrum, dominant_class, concentrations)
            - spectrum: Array of shape (n_bins,)
            - dominant_class: The class with highest concentration
            - concentrations: Array of shape (n_classes,)
        """
        n_classes = len(MATERIAL_CLASSES)
        
        if concentrations is not None:
            # Mixed spectrum with provided concentrations
            concentrations = np.asarray(concentrations)
            if len(concentrations) != n_classes:
                raise ValueError(f"concentrations must have length {n_classes}")
        elif class_id is not None:
            # Single class spectrum
            concentrations = np.zeros(n_classes)
            concentrations[class_id] = 1.0
        else:
            # Random single class
            class_id = self.rng.integers(0, n_classes)
            concentrations = np.zeros(n_classes)
            concentrations[class_id] = 1.0
        
        # Generate spectrum as weighted sum of class spectra
        spectrum = np.zeros(self.n_bins)
        for cid, conc in enumerate(concentrations):
            if conc > 0:
                spectrum += self.generate_single_class_spectrum(cid, conc)
        
        # Add baseline
        if add_baseline:
            spectrum += self._generate_baseline()
        
        # Add noise
        if add_noise:
            spectrum = self._add_noise(spectrum)
        
        # Ensure non-negative
        spectrum = np.clip(spectrum, 0, None)
        
        # Normalize to [0, 1]
        if normalize and spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        # Determine dominant class
        dominant_class = int(np.argmax(concentrations))
        
        return spectrum, dominant_class, concentrations
    
    def generate_mixed_spectrum(
        self,
        n_components: int = 2,
        add_baseline: bool = True,
        add_noise: bool = True,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Generate a mixed spectrum with random components.
        
        Args:
            n_components: Number of material components (2-5)
            add_baseline: Whether to add baseline continuum
            add_noise: Whether to add noise
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Tuple of (spectrum, dominant_class, concentrations)
        """
        n_classes = len(MATERIAL_CLASSES)
        n_components = min(n_components, n_classes)
        
        # Select random classes to mix
        selected_classes = self.rng.choice(n_classes, size=n_components, replace=False)
        
        # Generate random weights that sum to 1
        weights = self.rng.dirichlet(np.ones(n_components))
        
        # Build concentration vector
        concentrations = np.zeros(n_classes)
        concentrations[selected_classes] = weights
        
        return self.generate_spectrum(
            concentrations=concentrations,
            add_baseline=add_baseline,
            add_noise=add_noise,
            normalize=normalize,
        )
    
    def generate_dataset(
        self,
        n_samples: int,
        mixed_ratio: float = 0.2,
        n_components_range: Tuple[int, int] = (2, 3),
        return_labels: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate a dataset of synthetic LIBS spectra.
        
        Args:
            n_samples: Number of samples to generate
            mixed_ratio: Fraction of samples that are mixtures
            n_components_range: Range of components for mixed samples
            return_labels: Whether to return labels and concentrations
            
        Returns:
            If return_labels is False:
                spectra: Array of shape (n_samples, n_bins)
            If return_labels is True:
                Tuple of (spectra, labels, concentrations)
                - spectra: Array of shape (n_samples, n_bins)
                - labels: Array of shape (n_samples,) with dominant class
                - concentrations: Array of shape (n_samples, n_classes)
        """
        n_classes = len(MATERIAL_CLASSES)
        spectra = np.zeros((n_samples, self.n_bins))
        labels = np.zeros(n_samples, dtype=np.int64)
        concentrations = np.zeros((n_samples, n_classes))
        
        n_mixed = int(n_samples * mixed_ratio)
        n_pure = n_samples - n_mixed
        
        # Generate pure samples (evenly distributed among classes)
        samples_per_class = n_pure // n_classes
        idx = 0
        for class_id in range(n_classes):
            for _ in range(samples_per_class):
                if idx >= n_pure:
                    break
                spectrum, label, conc = self.generate_spectrum(class_id=class_id)
                spectra[idx] = spectrum
                labels[idx] = label
                concentrations[idx] = conc
                idx += 1
        
        # Fill remaining pure samples with random classes
        while idx < n_pure:
            spectrum, label, conc = self.generate_spectrum()
            spectra[idx] = spectrum
            labels[idx] = label
            concentrations[idx] = conc
            idx += 1
        
        # Generate mixed samples
        for i in range(n_pure, n_samples):
            n_components = self.rng.integers(
                n_components_range[0], 
                n_components_range[1] + 1
            )
            spectrum, label, conc = self.generate_mixed_spectrum(n_components=n_components)
            spectra[i] = spectrum
            labels[i] = label
            concentrations[i] = conc
        
        # Shuffle the dataset
        shuffle_idx = self.rng.permutation(n_samples)
        spectra = spectra[shuffle_idx]
        labels = labels[shuffle_idx]
        concentrations = concentrations[shuffle_idx]
        
        if return_labels:
            return spectra, labels, concentrations
        return spectra
    
    @staticmethod
    def get_class_info() -> Dict[int, Dict]:
        """Get information about material classes."""
        return {
            cid: {
                "name": material.name,
                "description": material.description,
                "n_peaks": len(material.peaks),
                "peak_positions": [p.position for p in material.peaks],
            }
            for cid, material in MATERIAL_CLASSES.items()
        }


if __name__ == "__main__":
    # Test the generator
    generator = SyntheticLIBSGenerator(seed=42)
    
    print("Material Classes:")
    for cid, info in generator.get_class_info().items():
        print(f"  {cid}: {info['name']} - {info['n_peaks']} peaks at {info['peak_positions']}")
    
    # Generate a small dataset
    spectra, labels, concentrations = generator.generate_dataset(
        n_samples=100,
        mixed_ratio=0.2,
    )
    
    print(f"\nGenerated dataset:")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Concentrations shape: {concentrations.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    print(f"  Spectrum range: [{spectra.min():.4f}, {spectra.max():.4f}]")
