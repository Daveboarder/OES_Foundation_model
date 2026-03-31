# LIBS Foundation Model - Architecture Documentation

This document describes the architecture, masking strategies, and design decisions for the LIBS Foundation Model.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Masking Strategies](#masking-strategies)
4. [Training Configurations](#training-configurations)
5. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

The LIBS Foundation Model is a self-supervised transformer for Laser Induced Breakdown Spectroscopy. The model learns spectral representations through **Masked Intensity Prediction (MIP)** — predicting masked regions of the spectrum from context.

### Key Design Principles

1. **Peak-biased masking**: Ensures the model learns peak structure, not just noise
2. **Variable block sizes**: Prevents the model from exploiting fixed masking patterns
3. **Contiguous masking**: Forces global reasoning over local interpolation
4. **Scalable architecture**: From 3M (local) to 500M+ (A100) parameters

---

## Model Architecture

### Embedding (input → residual stream)

```
Input Spectrum [batch, 2048] — raw intensity scalars, normalized to [0,1]
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Intensity Projection (learned 2-layer MLP)          │
│  Each scalar independently:                          │
│    Linear(1 → 128) → GELU → Linear(128 → 256)       │
│  Same weights shared across all 2048 bins            │
│  Output: [batch, 2048, 256]                          │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Mask Token Replacement (pretraining only)           │
│  80% of masked bins: projected vector REPLACED with  │
│    a single learned mask token (one 256-dim vector,  │
│    same for all positions — differentiated only by   │
│    positional encoding added below)                  │
│  10%: keeps projected random intensity               │
│  10%: keeps projected original value                 │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Prepend [CLS] Token                                 │
│  Learned 256-dim vector at position 0                │
│  Sequence: [CLS, bin_0, ..., bin_2047] → [2049, 256] │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  + Sinusoidal Positional Encoding (fixed)            │
│  Added element-wise — the only thing differentiating │
│  masked positions from each other                    │
└──────────────────────────────────────────────────────┘
    │
    ▼
  LayerNorm → this is x₀, the initial residual stream
```

### Residual Stream

The core of the transformer is a single stream of vectors that flows straight from
embedding to output. Nothing overwrites it — each sub-layer reads from the stream,
computes a delta, and **adds** it back. The stream accumulates information.

You can tap the stream at any point (after any layer) and run it through the
de-embedding head (MIP or classification) to get a valid, if less refined, output.
Early layers produce coarser predictions; later layers refine them. The final output
is the sum of the initial embedding and all deltas written by every attention and FFN
block.

```
x₀  (after embedding + pos enc + layernorm)
 │
 │         ┌─────────────────────────────────────────────────────────┐
 ├────────►│  Self-Attention (layer 1)                               │
 │         │  Every token attends to all 2049 tokens (no masking     │
 │         │  in the attention map — masked bins participate fully). │
 │◄────────│  Masked positions gather context from neighbors.        │
 │  += δ   └─────────────────────────────────────────────────────────┘
 │
x₁ = x₀ + δ_attn1
 │
 │         ┌─────────────────────────────────────────────────────────┐
 ├────────►│  FFN (layer 1)                                          │
 │         │  Linear(256→1024) → GELU → Linear(1024→256)             │
 │◄────────│  Per-position "thinking" — no cross-talk between tokens │
 │  += δ   └─────────────────────────────────────────────────────────┘
 │
x₂ = x₁ + δ_ffn1
 │
 │         ┌──────────────────────────────┐
 ├────────►│  Self-Attention (layer 2)    │
 │◄────────│                              │
 │  += δ   └──────────────────────────────┘
 │
 │         ┌──────────────────────────────┐
 ├────────►│  FFN (layer 2)              │
 │◄────────│                              │
 │  += δ   └──────────────────────────────┘
 │
 ┆          ... layers 3–5 ...
 │
 │         ┌──────────────────────────────┐
 ├────────►│  Self-Attention (layer 6)    │
 │◄────────│                              │
 │  += δ   └──────────────────────────────┘
 │
 │         ┌──────────────────────────────┐
 ├────────►│  FFN (layer 6)              │
 │◄────────│                              │
 │  += δ   └──────────────────────────────┘
 │
x₁₂ = x₀ + δ_attn1 + δ_ffn1 + δ_attn2 + ... + δ_ffn6
 │
 ▼
Final LayerNorm
 │
 ├──── position 0 ──► [CLS] embedding [256] ──► Classification / Regression head
 │                     (global spectrum summary,     (downstream tasks)
 │                      built entirely through
 │                      attention over 6 layers)
 │
 └──── positions 1–2048 ──► MIP head ──► predicted intensity per bin
                             (pretraining: loss on masked positions only)
```

**Key points:**

- The stream is the straight vertical line. It is never overwritten, only added to.
- At a masked position, x₀ contains no intensity info (just mask token + position).
  Every delta from attention writes in context gathered from unmasked neighbors.
  By x₁₂ the representation is built entirely from accumulated context.
- At an unmasked position, x₀ already contains the real intensity. The deltas enrich
  it with global spectral context, but the original info persists in the stream.
- You could run any intermediate x_i through the MIP head and get a prediction —
  it would just be less refined than using x₁₂. The model gradually builds up its
  representation; no single layer is solely responsible for the output.

### Model Configurations

| Config | d_model | n_heads | n_layers | d_ff | Parameters |
|--------|---------|---------|----------|------|------------|
| Local (16GB) | 256 | 8 | 6 | 1024 | ~3M |
| Standard | 256 | 8 | 6 | 1024 | ~3M |
| A100 Large | 1024 | 16 | 24 | 4096 | ~500M |

### Parameter Count Formula

```
params ≈ n_layers × (4 × d_model² + 2 × d_model × d_ff)
       + embeddings + heads
```

---

## Masking Strategies

The masking strategy is crucial for learning useful representations. We implement several options:

### 1. Random Scattered Masking (BERT-style)

```yaml
contiguous_masking: false
```

- Masks individual bins scattered randomly
- Simple but may allow local interpolation
- Not recommended for sparse spectral data

### 2. Contiguous Block Masking

```yaml
contiguous_masking: true
block_sizes: [50]  # Fixed size
```

- Masks contiguous regions of fixed size
- Forces the model to use global context
- Better than random for structured data

### 3. Variable Block Size Masking (Recommended)

```yaml
contiguous_masking: true
block_sizes: [25, 50, 100, 150]
```

- Block size randomly sampled from list
- Prevents model from exploiting fixed patterns
- Creates diverse training signal

### 4. Peak-Biased Masking (Recommended)

```yaml
peak_bias_enabled: true
peak_bias_ratio: 0.5      # 50% of masks must cover peaks
peak_threshold: 0.2       # Intensity > 0.2 = peak region
```

**Why Peak-Biased?**

LIBS spectra are sparse — most bins are noise/baseline (~95%). Random masking primarily masks noise, making the task trivial. Peak-biased masking ensures:

- At least `peak_bias_ratio` of masked bins overlap with peaks
- Model must learn peak structure, not just "predict low values"
- More challenging and informative training signal

**Peak Detection Algorithm:**

```python
# Simple threshold-based detection
peak_mask = spectrum > peak_threshold

# Expand to include peak shoulders (±5 bins)
for each peak position:
    mark positions [peak-5 : peak+5] as peak region
```

### Masking Statistics

The training script logs masking statistics:

```
Masking Statistics (from 100 samples):
  Avg masked bins: 307.2
  Avg masked ratio: 15.00%
  Peak coverage: 52.3%      # % of masks on peaks
  Avg peak bins: 156.8      # Avg bins with intensity > threshold
```

---

## Training Configurations

### Configuration Files

| File | Use Case | GPU Memory | Model Size |
|------|----------|------------|------------|
| `config_local.yaml` | Local testing | 8-16 GB | 3M |
| `config.yaml` | Standard training | 24-40 GB | 3M |
| `config_a100.yaml` | Large-scale | 40-80 GB | 500M |

### Key Parameters

#### Masking Parameters

```yaml
pretrain:
  mask_ratio: 0.15              # Total fraction masked
  mask_token_prob: 0.8          # % replaced with learnable mask embedding
  random_token_prob: 0.1        # % replaced with random intensity value
  # Remaining 10% kept unchanged (original intensity)

  contiguous_masking: true
  block_sizes: [25, 50, 100, 150]

  peak_bias_enabled: true
  peak_bias_ratio: 0.5
  peak_threshold: 0.2
```

#### Training Parameters

```yaml
pretrain:
  batch_size: 32-64
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_epochs: 10
  min_lr: 1e-6
```

### Dataset Sizes

| Config | Train | Validation | Test |
|--------|-------|------------|------|
| Local | 5,000 | 500 | 500 |
| Standard | 50,000 | 5,000 | 5,000 |
| A100 | 500,000 | 50,000 | 50,000 |

---

## Evaluation Metrics

### Pre-training (MIP)

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error on masked bins |
| RMSE | Root MSE |
| MAE | Mean absolute error on masked bins |
| R² | Coefficient of determination |
| Peak Coverage | % of masks that covered peaks |

### Fine-tuning (Classification)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Balanced Accuracy | Macro-averaged per-class accuracy |
| F1 (macro) | Macro F1 score |
| Confusion Matrix | Per-class prediction analysis |

### Fine-tuning (Regression)

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error |
| MAE | Mean absolute error |
| R² | Per-element R² scores |

---

## How BERT-style Masking Works for Continuous Spectra

In standard BERT (discrete tokens), 15% of tokens are selected for prediction and
the input is modified using the 80/10/10 rule. For continuous spectral intensities,
we implement this at two levels:

### Step 1: Dataset-level input modification (`MaskedLIBSDataset`)

The dataset selects 15% of bins (using contiguous + peak-biased strategy) and
modifies the input spectrum:

- **80% (type 1):** Intensity set to 0 (placeholder — will be replaced by mask embedding)
- **10% (type 2):** Intensity replaced with a random value in [0, 1]
- **10% (type 3):** Intensity kept unchanged

### Step 2: Embedding-level mask token insertion (`SpectralEmbedding`)

The embedding layer projects all intensity values to `d_model` vectors, then
replaces **only the type-1 positions** (the 80%) with a learnable mask token
embedding. Type-2 and type-3 positions keep their projected intensity embeddings.

```
Type 1 (80%): intensity=0 → project → REPLACED with learnable [MASK] embedding
Type 2 (10%): intensity=random → project → keeps projected random embedding
Type 3 (10%): intensity=original → project → keeps projected original embedding
```

### Step 3: Loss computation

MSE loss is computed on **all 15%** of masked positions (types 1, 2, and 3),
comparing the MIP head predictions against the original (unmasked) intensities.

### Why does this matter?

The 10% random + 10% unchanged strategy prevents the model from learning a
shortcut like "predict only when I see [MASK]." Since the model must also
correctly predict at positions where it sees random or original values, it is
forced to learn genuine spectral context — understanding that certain emission
lines co-occur, that peak ratios imply specific compositions, etc. This
produces representations that transfer well to downstream tasks where no masking
is applied.

---

## Design Decisions & Rationale

### Why Contiguous + Peak-Biased Masking?

1. **Sparse spectra problem**: ~95% of bins are noise. Random masking tests "predict noise" which is trivial.

2. **Local interpolation problem**: If individual bins are masked, the model can just average neighbors. Contiguous blocks force global reasoning.

3. **Variable sizes prevent overfitting**: Fixed 50-bin blocks could lead to pattern memorization. Variable sizes (25-150) create diverse training.

### Why Not Higher Mask Ratios?

We use 15% (BERT default), but for sparse data, higher ratios (30-40%) may work better. This is configurable:

```yaml
mask_ratio: 0.30  # 30% masking
```

### Why Transformer over CNN?

1. **Global context**: Attention can relate distant wavelengths (element correlations)
2. **Position flexibility**: Positional encoding handles wavelength information
3. **Transfer learning**: Transformer embeddings transfer well to downstream tasks

---

## Usage Examples

### Running Pre-training

```bash
# Local testing (quick)
uv run python train_pretrain.py \
    --config config/config_local.yaml \
    --experiment_name local_test

# A100 full training
uv run python train_pretrain.py \
    --config config/config_a100.yaml \
    --experiment_name a100_v1
```

### Checking Masking Statistics

```python
from data.dataset import MaskedLIBSDataset
from data.synthetic_generator import SyntheticLIBSGenerator

# Generate data
gen = SyntheticLIBSGenerator(seed=42)
spectra, _, _ = gen.generate_dataset(n_samples=1000)

# Create dataset with peak-biased masking
dataset = MaskedLIBSDataset(
    spectra=spectra,
    mask_ratio=0.15,
    contiguous_masking=True,
    block_sizes=[25, 50, 100, 150],
    peak_bias_enabled=True,
    peak_bias_ratio=0.5,
)

# Check statistics
stats = dataset.get_masking_stats(n_samples=100)
print(f"Peak coverage: {stats['peak_coverage']:.1%}")
```

---

## Future Improvements

1. **Curriculum masking**: Start with low mask ratio, increase over training
2. **Adaptive peak detection**: Learn peak threshold from data
3. **Contrastive pre-training**: Add SimCLR-style objective
4. **Multi-scale masking**: Mask at different resolutions

---

## References

- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (Joshi et al., 2020)
- DreaMS: Deep Representations for Mass Spectrometry (inspiration for spectral transformers)
