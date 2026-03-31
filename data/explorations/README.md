# Data Explorations

Scripts for exploring and visualizing the synthetic LIBS data.

## Usage

```bash
# From project root
uv run python data/explorations/visualize_synthetic_data.py
```

## Generated Figures

The script generates the following visualizations in `figures/`:

1. **01_all_classes.png** - Example spectra for all 5 material classes
2. **02_class_comparison.png** - All classes overlaid (no noise/baseline)
3. **03_noise_effect.png** - Effect of noise and baseline on spectra
4. **04_intra_class_variation.png** - Variation within same class
5. **05_mixed_spectra.png** - Examples of mixed material spectra
6. **06_peak_positions.png** - Peak positions by material class
7. **07_dataset_statistics.png** - Dataset statistics and distributions
8. **08_masking_preview.png** - Preview of masking for pre-training
