# LIBS Foundation Model - Architecture Documentation

This document describes the architecture, masking strategies, and design decisions for the LIBS Foundation Model.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Sources](#data-sources)
4. [Masking Strategies](#masking-strategies)
5. [Downstream Fine-Tuning](#downstream-fine-tuning)
6. [Training Configurations](#training-configurations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage Examples](#usage-examples)

---

## Overview

The LIBS Foundation Model is a self-supervised transformer for Laser Induced Breakdown Spectroscopy. The model learns spectral representations through **Masked Intensity Prediction (MIP)** — predicting masked regions of the spectrum from context.

### Key Design Principles

1. **Peak-biased masking**: Ensures the model learns peak structure, not just noise
2. **Variable block sizes**: Prevents the model from exploiting fixed masking patterns
3. **Contiguous masking**: Forces global reasoning over local interpolation
4. **Scalable architecture**: Encoder size scales with `d_model` and `n_layers`; sequence length is either **17,428 bins** (intensity mode) or **thousands of line tokens** (line-token mode)

### Data and embedding modes

All production training uses the **physics pipeline** via `--libs_data_config` (`config/libs_data.yaml`).

| Embedding | `model.embedding_type` | Sequence | Pre-train target |
|-----------|------------------------|----------|------------------|
| **Bin (intensity)** | `intensity` (default) | ~17,428 bins + CLS | Masked bin intensities (MIP) |
| **Line-token** | `line_token` | `n_lines` + CLS (threshold-dependent) | Masked Voigt features (`max_I`, `FWHM`) |
| **Line-token-linear** | `line_token_linear` | `n_lines` + CLS | Same MIP targets; reads `line_tokens_*.h5` |

Line-token modes require `--line_embedding_config` (`config/line_embedding.yaml`). When the flag is set and the model config does not pin `embedding_type`, training defaults to **`line_token_linear`**. Checkpoints are **not interchangeable** between modes.

`--libs_data_config` overrides `data.n_bins` (intensity) or drives line-feature caches (line-token), and sets `model.max_seq_len` to `n_bins + 1` or `n_lines + 1`. Fine-tune also overrides `n_classes`, `n_elements`, and `n_concentration_bins` from the libs data config.

---

## Model Architecture

### Transformer Encoder (high-level)

```
Intensity mode                    Line-token mode              Line-token-linear mode
──────────────                  ───────────────              ──────────────────────
Input [B, n_bins]               Input [B, L, 6] features       Input [B, L, 14] tokens
    │                               │                            │
    ▼                               ▼                            ▼
SpectralEmbedding               LineTokenEmbedding           LinearLineTokenEmbedding
MLP per bin                     runtime concat +             z-score + nn.Linear(14)
                                element/ion Embeddings       (pre-baked HDF5)
    │                               │                            │
    └───────────────────────────────┴────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────┐
│  + [CLS] Token (learned)                │
│  + Sinusoidal Positional Encoding       │
│  Seq len: n_bins+1  or  n_lines+1       │
│  (line mode: key_padding_mask invalid)  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder Blocks (× N)       │
│  - Multi-Head Self-Attention            │
│  - Feed-Forward Network                 │
│  - Pre-LayerNorm + Residual Connections │
└─────────────────────────────────────────┘
    │
    ├────────────────────┐
    ▼                    ▼
[CLS] Embedding     Sequence Output
    │                    │
    ▼                    ▼
Classification/     MIP Prediction
Regression Head     Head (pre-training)
```

### Embedding Details

The embedding pipeline transforms raw intensities into the initial residual stream:

1. **Intensity projection** — a learned 2-layer MLP (`Linear(1→d_model/2) → GELU → Linear(d_model/2→d_model)`)
   projects each scalar intensity independently to a d_model-dimensional vector. The same
   weights are shared across all `n_bins` positions.

2. **Mask token replacement** (pretraining only) — masked positions have their projected
   vectors replaced with a single learned mask token (one d_model-dimensional vector, shared across all
   masked positions). See [How BERT-style Masking Works](#how-bert-style-masking-works-for-continuous-spectra)
   for the 80/10/10 replacement strategy.

3. **[CLS] token** — a learned d_model-dimensional vector prepended at position 0. It carries no spectral
   information initially; through attention across all layers it accumulates a global summary
   of the entire spectrum. Used for downstream classification/regression.

4. **Positional encoding** — fixed sinusoidal encoding added element-wise. This is the only
   thing differentiating masked positions from each other (they all share the same mask token).

5. **LayerNorm** — produces x₀, the initial state of the residual stream.

### Residual Stream Perspective (alternative view of the transformer)

The transformer can be understood as a single stream of vectors flowing straight from
embedding to output. Nothing overwrites it — each sub-layer reads from the stream,
computes a delta, and **adds** it back.

```
x₀  (embedding output — the initial stream)
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 1) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 1)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 2) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 2)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 ┆          ... layers 3–5 ...
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 6) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 6)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 ▼
x_final = x₀ + δ_attn1 + δ_ffn1 + ... + δ_ffn6
 │
 ▼
Final LayerNorm → output heads
```

**How to read this:**

- The straight vertical line **is** the residual stream — it is never overwritten, only added to.
- Self-attention reads all n_bins+1 positions, computes interactions between them, and writes a
  delta back. This is where tokens exchange information (masked positions gather context
  from unmasked neighbors).
- FFN processes each position independently — per-token "thinking" with no cross-talk.
- The stream can be tapped at **any intermediate point** and fed through the output head
  (MIP or classification) to get a valid prediction. Early taps give coarser results;
  later layers refine them. The final output is the sum of all deltas.
- At a masked position, x₀ has no intensity info — just mask token + positional encoding.
  The entire representation is built from context accumulated through attention deltas.
- At an unmasked position, x₀ already has the real intensity. Deltas enrich it with
  global spectral context.

### Model Configurations

| Config file | Embedding | d_model | n_layers | Use case |
|-------------|-----------|---------|----------|----------|
| `config_libs_smoke.yaml` | intensity | 32 | 1 | Pipeline wiring smoke |
| `config_libs_4090.yaml` | intensity | 128 | 4 | Bin embedding, RTX 4090 |
| `config_libs_a100.yaml` | intensity | 256 | 6 | Bin embedding, A100 |
| `config_libs_token.yaml` | line_token | 32 | 1 | Line-token smoke (+ `line_embedding_smoke.yaml`) |
| `config_libs_token_4090.yaml` | line_token | 128 | 4 | Line-token, RTX 4090 (+ `line_embedding.yaml`) |
| `config_libs_token_linear.yaml` | line_token_linear | 64 | 2 | Pre-baked tokens smoke |
| `config_libs_token_linear_4090.yaml` | line_token_linear | 128 | 4 | Pre-baked tokens, RTX 4090 |

At ~17k bins or ~8k lines, `LIBSTransformer` uses PyTorch SDPA (`need_weights=False`) for tractable bf16 attention on consumer GPUs.

### Parameter Count Formula

```
params ≈ n_layers × (4 × d_model² + 2 × d_model × d_ff)
       + embeddings + heads
```

---

## Data Sources

### Physics-based pipeline (`data/libs_pipeline.py` + `config/libs_data.yaml`)

- Voigt-profile synthesis using `external_data/Source/LIBS_data_vacuum.db`, concentration ranges from `Samples_Fe_matrix.xlsx`, and wavelength axis from `external_data/Data/VASKUT K8.json`.
- Materialises spectra once into `external_data/cache/synthetic_cache_{md5}.h5`; train/val/test splits in `splits_{md5}.json` are shared between pretrain and finetune.
- Downstream labels: per-element concentrations from the sample table; classification labels via K-means on concentration vectors (`cluster_compositions`, `n_clusters` from `libs_data.yaml`).
- Smoke variant: `config/libs_data_smoke.yaml` — 3 sample types × 6 shots per type.

### Runtime overrides (`--libs_data_config`)

| Field | New value | Scripts |
|-------|-----------|---------|
| `data.n_bins` | length of wavelength array | `train_pretrain.py`, `train_finetune.py` |
| `model.max_seq_len` | `n_bins + 1` (intensity) or `n_lines + 1` (line-token) | same |
| `data.n_classes` | `downstream.n_clusters` | `train_finetune.py` only |
| `data.n_elements` | count of `elements_to_predict` (or all ~60) | `train_finetune.py` only |
| `data.n_concentration_bins` | `downstream.n_concentration_bins` | `train_finetune.py` only |

`run_info.yaml` records `libs_data_config`, and `line_embedding_config` when applicable.

### Line-token assets (`--line_embedding_config`)

Three HDF5 caches are built in sequence (additive — intermediate files are kept for debugging):

| Step | Module | Output | Role |
|------|--------|--------|------|
| 1 | `data/line_dictionary.py` | `line_dict_{hash}.h5` | Theoretical intensities over Te×Ne; default selection keeps top 10% most intense lines per element (`min_keep: 10`, legacy threshold mode still supported) |
| 2 | `data/line_features.py` | `line_features_{hash}.h5` | Per-spectrum Voigt fits: `[n_spectra, n_lines, 6]` (`max_I`, `FWHM`, `R²`, `Δλ`, `RMSE`, `fit_valid`) |
| 3 | `data/line_tokenization.py` | `line_tokens_{hash}.h5` | **Separated tokenization**: merged 14-feature tensor per line, raw values + `feature_mean`/`feature_std` attrs |

**Standalone tokenization** (step 3 only, after 1–2 exist):

```bash
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml
```

Or all three via `prepare_line_tokens_assets()` (`data/line_embedding_pipeline.py`).

#### `line_tokens_*.h5` layout

```
tokens          [n_spectra, n_lines, 14]  float32   # raw features (z-scored at train time)
fit_valid       [n_spectra, n_lines]      uint8     # 1 = successful Voigt fit
central_wavelength [n_lines]             float32   # for sinusoidal PE
attrs: feature_names, feature_mean, feature_std, line_dict_hash, line_features_hash
```

**14 channels (indices):** 0 λ, 1 Ei, 2 Ek, 3–6 log10(gi/gk/Ak/I_theory), 7 atomic_number (Z), 8 ion_binary (0=I, 1=II+), 9 max_intensity, 10 fwhm, 11 r2, 12 delta_lambda, 13 rmse.

#### Embedding at training time

| Mode | Reads | Embedding module |
|------|-------|------------------|
| `line_token` | `line_features_*.h5` | `LineTokenEmbedding` — 7 quantum scalars + `nn.Embedding` element/ion + 5 fit features → 2-layer MLP |
| `line_token_linear` | `line_tokens_*.h5` only | `LinearLineTokenEmbedding` — z-score using stored stats → `nn.Linear(14, d_model)` |

Invalid Voigt fits use `key_padding_mask` from the separate `fit_valid` dataset (not mixed into the 14 feature channels).

Pre-training masks a fraction of valid lines and predicts masked `max_intensity` and `fwhm` (`MaskedLineTokenDataset` or `MaskedLineTokensDataset`, `training/pretrain.py`).

---

## Masking Strategies

Masking below applies to **intensity (bin) mode**. Line-token modes mask entire line tokens and zero the `max_intensity` / `fwhm` channels in the input (indices 9–10 in `line_tokens_*.h5`, or channels 0–1 in `line_features_*.h5`).

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

## Downstream Fine-Tuning

Fine-tuning attaches task-specific heads to the frozen or jointly trained `LIBSTransformer` encoder via `LIBSFinetuneModule` (`training/finetune.py`). Heads live in `models/heads.py`.

### Pooling (`--pool`)

The encoder returns `cls_embedding` and `sequence_embeddings`. Fine-tuning collapses them before the heads:

| Pool | Representation | Head input dim |
|------|----------------|----------------|
| `cls` | CLS token only | `d_model` |
| `mean` | Mean over wavelength bins (excluding CLS) | `d_model` |
| `cls_mean` | Concatenation of CLS and mean-pool | `2 × d_model` |

### Task modes (`--task`)

| Task | Head | Loss | Targets in batch |
|------|------|------|------------------|
| `classification` | `ClassificationHead` | Cross-entropy | `label` |
| `quantification` | `RegressionHead` (sigmoid) | MSE | `concentrations` |
| `quantification_binned` | `BinnedQuantificationHead` | Per-element CE on 1000 bins | `concentrations` (bins encoded on the fly) |
| `regression` | alias of `quantification` | MSE | `concentrations` |
| `both` | classification + regression | CE + MSE | `label` + `concentrations` |

`BinnedQuantificationHead` uses one small MLP branch per element, each outputting `n_concentration_bins` logits (default 1000). `concentration_to_bin` / `bin_to_concentration` in `models/heads.py` map between float targets in [0, 1] and bin indices.

### Optimizer split

AdamW with two parameter groups: encoder at `0.1 × learning_rate`, heads at full `learning_rate`; cosine schedule with linear warmup.

### Checkpointing

`ModelCheckpoint` monitors task-specific metrics: `val/accuracy` (classification), `val/reg_mae` (quantification), `val/bin_accuracy` (binned), or `val/loss` (`both`). Lightning checkpoints store `encoder.*` and head weights; `final_encoder.pt` is a raw encoder state dict for evaluation reload.

---

## Training Configurations

### Model / training YAML files

| File | Embedding | GPU | Pretrain epochs | Finetune epochs | Logger |
|------|-----------|-----|-----------------|-----------------|--------|
| `config_libs_smoke.yaml` | intensity | any | 2 | 2 | tensorboard |
| `config_libs_4090.yaml` | intensity | RTX 4090 | 5 | 5 | tensorboard |
| `config_libs_a100.yaml` | intensity | A100 | 60 | 30 | tensorboard |
| `config_libs_token.yaml` | line_token | any | 2 | 2 | tensorboard |
| `config_libs_token_4090.yaml` | line_token | RTX 4090 | 5 | 5 | tensorboard |
| `config_libs_token_linear.yaml` | line_token_linear | any | 2 | 2 | tensorboard |
| `config_libs_token_linear_4090.yaml` | line_token_linear | RTX 4090 | 5 | 5 | tensorboard |

Pair with `--libs_data_config config/libs_data.yaml`. Line-token configs also require `--line_embedding_config config/line_embedding.yaml` (or `line_embedding_smoke.yaml`).

### Data pipeline YAML files

| File | Sample types | Shots / type | Clusters | Binned bins |
|------|--------------|--------------|----------|-------------|
| `libs_data.yaml` | ~2256 (all) | 50 | 10 | 1000 |
| `libs_data_smoke.yaml` | 3 | 6 | 3 | 50 |

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
| `libs_data_smoke.yaml` | 18 | 4 | 4 |
| `libs_data.yaml` (cached) | ~79,030 | ~16,935 | ~16,935 | splits from `splits_{md5}.json` |

### SLURM deployment

| Script | Purpose |
|--------|---------|
| `run_pretrain_libs.slurm` | Bin-embedding pretrain (`config_libs_a100.yaml` + `libs_data.yaml`) |
| `run_finetune_libs.slurm` | Binned finetune (`quantification_binned`, `cls_mean`) |
| `run_eval.slurm` | Post-training evaluation |

For line-token runs, mirror these scripts with `config_libs_token_4090.yaml` (runtime embedding) or `config_libs_token_linear_4090.yaml` (pre-baked tokens) and `--line_embedding_config config/line_embedding.yaml`.

All SLURM jobs run in the `sslibs` enroot container with the workspace mounted at `/workspace`.

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

### Fine-tuning (Regression / quantification)

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error |
| MAE | Mean absolute error |
| R² | Per-element R² scores (averaged) |

### Fine-tuning (Binned quantification)

| Metric | Description |
|--------|-------------|
| `bin_accuracy` | Fraction of (sample, element) pairs with correct argmax bin |
| `decoded_mae` | MAE after mapping predicted bins back to [0, 1] concentrations |
| `decoded_r2` | R² on decoded concentrations — primary signal for whether the model beats the per-element mean |

`bin_accuracy` alone can be misleading when most elements are near-zero (predicting bin 0 scores well). Prefer `decoded_mae` and `decoded_r2` for model selection.

`evaluate_model.py` currently loads classification and regression heads only; binned runs should be judged from TensorBoard logs and `run_info.yaml` `test_results` until evaluation support is extended.

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

### Running pre-training

```bash
# Bin embedding — RTX 4090
uv run python train_pretrain.py \
    --config config/config_libs_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --experiment_name libs_bin_pretrain \
    --num_workers 4

# Line-token embedding — RTX 4090 (builds line caches on first run)
uv run python train_pretrain.py \
    --config config/config_libs_token_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --experiment_name libs_line_pretrain \
    --num_workers 4

# Line-token-linear — pre-baked tokens (optional standalone build first)
uv run python scripts/build_line_tokens.py \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml
uv run python train_pretrain.py \
    --config config/config_libs_token_linear_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --experiment_name libs_token_linear_pretrain \
    --num_workers 4

# A100 bin embedding via SLURM
sbatch run_pretrain_libs.slurm
```

### Running fine-tuning

```bash
# Binned quantification — bin embedding (match pretrain config)
uv run python train_finetune.py \
    --config config/config_libs_4090.yaml \
    --pretrain_run_dir runs/pretrain_<timestamp>_libs_bin_pretrain \
    --libs_data_config config/libs_data.yaml \
    --task quantification_binned \
    --pool cls_mean \
    --num_workers 4

# Binned quantification — line-token (match pretrain + line_embedding_config)
uv run python train_finetune.py \
    --config config/config_libs_token_4090.yaml \
    --pretrain_run_dir runs/pretrain_<timestamp>_libs_line_pretrain \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --task quantification_binned \
    --pool cls_mean

sbatch run_finetune_libs.slurm runs/pretrain_<timestamp>_libs_pretrain_a100
```

### Evaluation and run discovery

```bash
uv run python list_runs.py --latest
uv run python evaluate_model.py --run_dir runs/pretrain_<timestamp>_...
uv run python evaluate_model.py --run_dir runs/finetune_<timestamp>_... --mode finetune
```

Runs are stored under `runs/{type}_{timestamp}_{experiment}/` with `config.yaml`, `run_info.yaml`, `checkpoints/`, `logs/`, and optional `evaluation/`. See `PROJECT_SUMMARY.json` / `PROJECT_SUMMARY.html` for full layout and CLI reference.

### Checking masking statistics (bin mode)

Load cached spectra from the libs pipeline, wrap in `MaskedLIBSDataset`, and call `get_masking_stats(n_samples=100)` — see `PretrainDataModule.setup()` logging during training.

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
