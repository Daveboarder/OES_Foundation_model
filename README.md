# LIBS Foundation Model

A self-supervised foundation model for Laser Induced Breakdown Spectroscopy (LIBS) using a BERT-style transformer architecture with masked intensity prediction.

## Overview

This project implements a foundation model for LIBS spectroscopy that:

1. **Pre-trains** on unlabeled spectra using Masked Intensity Prediction (MIP) — predicting masked regions of the spectrum
2. **Fine-tunes** for downstream tasks: material classification and element concentration regression
3. Uses **synthetic data** with 5 material classes for development and testing

The key idea is that self-supervised pre-training learns useful spectral representations without labels, enabling better downstream performance with fewer labeled samples.

## Features

- **Self-supervised pre-training** via Masked Intensity Prediction (MIP)
- **Contiguous + peak-biased masking** — masks variable-size blocks (25-100 bins) biased toward peak regions, with BERT-style 80/10/10 token replacement
- **Transformer architecture** adapted for continuous spectral data (2048 bins)
- **Synthetic data generation** with 5 material classes (Fe, Cu, Al, Ca, Mixed)
- **Fine-tuning support** for classification and concentration regression
- **Organized run management** — each run creates a timestamped folder with all outputs
- **Comprehensive evaluation** with visualizations and metrics
- **Two configs**: full training (A100) and local testing (16GB GPU)

---

## Quick Start

### 1. Setup Environment

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `uv.lock`.

### 2. Explore Synthetic Data

```bash
uv run python data/explorations/visualize_synthetic_data.py
```

Generates visualizations in `data/explorations/figures/`.

### 3. Run Pre-training (Local Test)

```bash
uv run python train_pretrain.py --config config/config_local.yaml --experiment_name my_test
```

This creates a timestamped run directory like:
```
runs/pretrain_2024-02-05_14-30-25_my_test/
├── config.yaml          # Copy of config used
├── run_info.yaml        # Run metadata
├── checkpoints/         # Model checkpoints
│   ├── epoch=05-val_loss=0.0150.ckpt
│   ├── last.ckpt
│   └── final_model.pt
└── logs/                # TensorBoard logs
```

### 4. Evaluate the Model

```bash
# Using run directory (recommended)
uv run python evaluate_model.py --run_dir runs/pretrain_2024-02-05_14-30-25_my_test

# Or with explicit paths (legacy)
uv run python evaluate_model.py \
    --checkpoint runs/pretrain_2024-02-05_14-30-25_my_test/checkpoints/final_model.pt \
    --config config/config_local.yaml \
    --mode pretrain
```

Evaluation outputs go to `{run_dir}/evaluation/`.

### 5. List All Runs

```bash
uv run python list_runs.py                    # List all runs
uv run python list_runs.py --type pretrain    # Filter by type
uv run python list_runs.py --latest           # Show most recent
uv run python list_runs.py --details runs/pretrain_...  # Show details
```

---

## Run Directory Structure

Each training run creates a self-contained folder with everything needed:

```
runs/
├── pretrain_2024-02-05_14-30-25_my_experiment/
│   ├── config.yaml          # Exact config used (for reproducibility)
│   ├── run_info.yaml        # Metadata: params, samples, status, etc.
│   ├── checkpoints/
│   │   ├── epoch=05-val_loss=0.0150.ckpt
│   │   ├── last.ckpt
│   │   └── final_model.pt
│   ├── logs/                # TensorBoard or W&B logs
│   └── evaluation/          # Created by evaluate_model.py
│       └── eval_2024-02-05_16-00-00/
│           ├── pretrain_reconstruction_examples.png
│           ├── pretrain_error_analysis.png
│           └── pretrain_metrics.txt
│
├── finetune_2024-02-06_09-00-00_classification/
│   ├── config.yaml
│   ├── run_info.yaml        # Includes reference to pretrain run
│   ├── checkpoints/
│   └── evaluation/
```

---

## Configuration

Three configuration files are provided:

| Config | Use Case | GPU | Model Size | Data Size | Epochs |
|--------|----------|-----|------------|-----------|--------|
| `config/config_local.yaml` | Local testing | RTX 16GB | 3M params | 5k samples | 10 |
| `config/config.yaml` | Standard training | 24-40GB | 3M params | 50k samples | 100 |
| `config/config_a100.yaml` | Large-scale | A100 40-80GB | **500M params** | 500k samples | 100 |

### Key Configuration Options

```yaml
# config/config.yaml (or config_local.yaml)

data:
  n_bins: 2048                    # Spectrum resolution
  n_classes: 5                    # Number of material classes
  synthetic:
    train_samples: 50000          # Training data size
    noise_sigma: 0.02             # Noise level

model:
  d_model: 256                    # Transformer hidden dimension (1024 for A100)
  n_heads: 8                      # Attention heads (16 for A100)
  n_layers: 6                     # Transformer layers (24 for A100)
  d_ff: 1024                      # Feed-forward dimension (4096 for A100)
  dropout: 0.1                    # Dropout rate

pretrain:
  mask_ratio: 0.15                # Fraction of spectrum to mask
  
  # Contiguous block masking (recommended)
  contiguous_masking: true
  block_sizes: [25, 50, 100, 150] # Variable block sizes (randomly sampled)
  
  # Peak-biased masking (ensures masks cover peaks, not just noise)
  peak_bias_enabled: true
  peak_bias_ratio: 0.5            # 50% of masks must overlap with peaks
  peak_threshold: 0.2             # Intensity threshold for peak detection
  
  batch_size: 64
  epochs: 100
  learning_rate: 1.0e-4

finetune:
  batch_size: 32
  epochs: 50
  freeze_encoder: false           # Whether to freeze pre-trained weights

logging:
  logger: "tensorboard"           # "tensorboard" or "wandb"

device:
  precision: "bf16-mixed"         # "bf16-mixed" (A100), 16 (older GPUs), 32 (full)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation on masking strategies and model design.

---

## Training Pipeline

### Step 1: Pre-training (Self-Supervised)

Pre-train the transformer using Masked Intensity Prediction:

```bash
# Full training (A100)
uv run python train_pretrain.py --config config/config.yaml --experiment_name pretrain_v1

# Local testing (16GB GPU)
uv run python train_pretrain.py --config config/config_local.yaml --experiment_name local_test
```

**Options:**
| Flag | Description |
|------|-------------|
| `--config PATH` | Config file to use |
| `--runs_dir DIR` | Base directory for runs (default: `runs/`) |
| `--experiment_name NAME` | Name added to run folder |
| `--save_data` | Save generated data to run directory |
| `--early_stopping` | Enable early stopping on val loss |
| `--num_workers N` | Data loading workers (default: 0) |
| `--seed N` | Random seed (default: 42) |

**Output:** `runs/pretrain_{timestamp}_{experiment_name}/`

### Step 2: Fine-tuning (Supervised)

Fine-tune the pre-trained model for downstream tasks:

```bash
# Using pretrain run directory (recommended)
uv run python train_finetune.py \
    --pretrain_run_dir runs/pretrain_2024-02-05_14-30-25_my_test \
    --task both \
    --experiment_name finetune_v1

# Or with explicit checkpoint path
uv run python train_finetune.py \
    --pretrained_checkpoint path/to/model.pt \
    --config config/config.yaml \
    --task both
```

**Options:**
| Flag | Description |
|------|-------------|
| `--pretrain_run_dir PATH` | Pretrain run directory (auto-finds checkpoint) |
| `--pretrained_checkpoint PATH` | Path to pre-trained model (legacy) |
| `--task {classification,regression,both}` | Downstream task |
| `--freeze_encoder` | Freeze encoder weights (linear probe) |

**Output:** `runs/finetune_{timestamp}_{task}/`

### Step 3: Evaluation

Evaluate trained models with comprehensive visualizations:

```bash
# Evaluate using run directory (recommended)
uv run python evaluate_model.py --run_dir runs/pretrain_2024-02-05_14-30-25_my_test

# Auto-detects mode (pretrain/finetune) from run directory
```

**Outputs (saved to `{run_dir}/evaluation/`):**
- `pretrain_reconstruction_examples.png` — Original vs reconstructed spectra
- `pretrain_error_analysis.png` — Error distribution and analysis
- `pretrain_embeddings_tsne.png` — t-SNE of learned representations
- `finetune_classification.png` — Confusion matrix and accuracy
- `finetune_regression.png` — Scatter plots for concentration prediction
- `*_metrics.txt` — Numeric metrics summary

---

## Data Exploration

Visualize synthetic data characteristics:

```bash
uv run python data/explorations/visualize_synthetic_data.py
```

**Generated plots (`data/explorations/figures/`):**

| File | Description |
|------|-------------|
| `01_all_classes.png` | Example spectra for all 5 material classes |
| `02_class_comparison.png` | All classes overlaid (no noise) |
| `03_noise_effect.png` | Effect of noise and baseline |
| `04_intra_class_variation.png` | Variation within same class |
| `05_mixed_spectra.png` | Mixed material spectra |
| `06_peak_positions.png` | Peak positions by class |
| `07_dataset_statistics.png` | Dataset statistics |
| `08_masking_preview.png` | Contiguous masking visualization |

---

## Project Structure

```
self_supervised_LIBS/
├── config/
│   ├── config.yaml              # Full training config
│   ├── config_local.yaml        # Local testing config (16GB GPU)
│   └── config_a100.yaml         # Large-scale A100 config
├── data/
│   ├── synthetic_generator.py   # Synthetic LIBS data generation
│   ├── dataset.py               # PyTorch Dataset classes
│   └── explorations/            # Data visualization scripts
├── models/
│   ├── libs_transformer.py      # Main transformer architecture
│   ├── positional_encoding.py   # Positional encoding variants
│   └── heads.py                 # Classification/regression heads
├── training/
│   ├── pretrain.py              # Pre-training Lightning module
│   └── finetune.py              # Fine-tuning Lightning module
├── utils/
│   ├── metrics.py               # Evaluation metrics and plotting
│   └── run_manager.py           # Run directory management
├── runs/                        # [Created] Run directories
│   ├── pretrain_2024-02-05.../
│   └── finetune_2024-02-06.../
├── train_pretrain.py            # Pre-training entry point
├── train_finetune.py            # Fine-tuning entry point
├── evaluate_model.py            # Post-training evaluation
├── list_runs.py                 # List and inspect runs
└── pyproject.toml               # Project configuration
```

---

## Model Architecture

```
Input: 2048-bin spectrum (normalized to [0,1])
   │
   ▼
┌─────────────────────────────────┐
│  Linear Embedding (1 → 256)    │
│  + Sinusoidal Positional Enc   │
│  + [CLS] Token                 │
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  Transformer Encoder           │
│  - 6 layers                    │
│  - 8 attention heads           │
│  - 256 hidden dim              │
│  - 1024 FFN dim                │
└─────────────────────────────────┘
   │
   ├──────────────────┐
   ▼                  ▼
[CLS] Token      Sequence Output
   │                  │
   ▼                  ▼
Classification    MIP Prediction
/ Regression      (pre-training)
```

**Pre-training objective**: Masked Intensity Prediction (MIP)
- Mask 15% of bins in contiguous blocks (variable sizes: 25-100 bins), biased toward peak regions
- BERT-style 80/10/10: 80% get learnable mask embedding, 10% random value, 10% unchanged
- Model predicts original intensities at all masked positions (MSE loss)
- Forces the model to learn spectral context: if it sees certain lines, it must predict others

**Downstream tasks**:
- Classification: Material type (5 classes)
- Regression: Element concentrations (5 values, sum to 1)

---

## Logging

### TensorBoard (default)

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

### Weights & Biases

Edit config to enable:
```yaml
logging:
  logger: "wandb"
  wandb:
    project: "libs-foundation-model"
    entity: null  # Your username or team
```

Then login:
```bash
wandb login
```

---

## Hardware Requirements

| Config | GPU VRAM | System RAM | Notes |
|--------|----------|------------|-------|
| `config_local.yaml` | 8-16 GB | 16 GB | RTX 3080/4060 Ti |
| `config.yaml` | 40+ GB | 32 GB | A100/H100 |

**CUDA versions supported**: 11.8, 12.1, 12.4

---

## Typical Workflow

```bash
# 1. Setup
uv sync

# 2. Explore data
uv run python data/explorations/visualize_synthetic_data.py

# 3. Quick local test
uv run python train_pretrain.py --config config/config_local.yaml --experiment_name test1

# 4. List runs
uv run python list_runs.py --latest

# 5. Evaluate (uses latest run)
uv run python evaluate_model.py --run_dir runs/pretrain_...

# 6. Fine-tune from the pretrain run
uv run python train_finetune.py \
    --pretrain_run_dir runs/pretrain_... \
    --task both

# 7. Evaluate fine-tuned model
uv run python evaluate_model.py --run_dir runs/finetune_...

# 8. Full training on A100
uv run python train_pretrain.py --config config/config_a100.yaml --experiment_name full_v1
```

---

## SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=libs_pretrain
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# Inside container
uv run python train_pretrain.py \
    --config config/config_a100.yaml \
    --experiment_name slurm_run
```

---

## License

MIT
