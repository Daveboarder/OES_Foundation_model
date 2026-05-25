# LIBS Foundation Model

A self-supervised foundation model for Laser Induced Breakdown Spectroscopy (LIBS) using a BERT-style transformer. Training uses the **physics-based LIBS pipeline** (Voigt synthesis, ~17k wavelength bins, HDF5 cache) with a choice of **bin (intensity) embedding** or **line-token embedding**.

## Overview

1. **Pre-trains** on unlabeled spectra (Masked Intensity Prediction on bins, or masked line-feature prediction on spectral lines)
2. **Fine-tunes** for classification, regression, and **binned quantification** (`quantification_binned` with decoded R²)
3. **Data** — `config/libs_data.yaml` (~112k shots from the sample matrix) or `config/libs_data_smoke.yaml` for quick tests

See [ARCHITECTURE.md](ARCHITECTURE.md) for design detail, [PROJECT_SUMMARY.html](PROJECT_SUMMARY.html) / [PROJECT_SUMMARY.json](PROJECT_SUMMARY.json) for full CLI and layout reference.

## Embedding modes

| Mode | Config | CLI | Sequence | Pre-train target |
|------|--------|-----|----------|------------------|
| **Bin (intensity)** | `embedding_type: intensity` (default) | `--libs_data_config` only | ~17,428 bins + CLS | Masked bin intensities (MIP) |
| **Line-token** | `embedding_type: line_token` | `--libs_data_config` + `--line_embedding_config` | ~2k–8k lines + CLS | Masked Voigt features (max_I, FWHM) |
| **Line-token-linear** | `embedding_type: line_token_linear` | same CLI (default when `--line_embedding_config` is set) | ~2k–8k lines + CLS | Same MIP targets; reads pre-baked tokens |

Checkpoints are **not interchangeable** between modes (different sequence length and weights).

### Line-token pipeline (three HDF5 caches)

Offline preprocessing (reusable across training runs):

1. **Line dictionary** (`data/line_dictionary.py`) — theoretical intensities over a Te×Ne grid from `LIBS_data_vacuum.db`; keep lines above `intensity_threshold` → `line_dict_*.h5`
2. **Line features** (`data/line_features.py`) — per-spectrum Voigt fit at each line centre → `line_features_*.h5` `[n_spectra, n_lines, 6]`
3. **Line tokens** (`data/line_tokenization.py`) — merge dictionary + fits into one tensor per line → `line_tokens_*.h5` `[n_spectra, n_lines, 14]` (raw values + `feature_mean`/`feature_std` in attrs)

At training time:

- **`line_token`** — loads `line_features_*.h5`; `LineTokenEmbedding` concatenates quantum scalars + `nn.Embedding` element/ion + fit features at runtime.
- **`line_token_linear`** — loads only `line_tokens_*.h5`; `LinearLineTokenEmbedding` z-scores the 14 channels and applies `nn.Linear(14, d_model)`.

Build tokens once:

```bash
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml
```

**Token features (14 channels, stored raw):** wavelength, Ei, Ek, log10(gi/gk/Ak/I_theory), atomic number Z, ion_binary (0=I, 1=II+), max_intensity, FWHM, R², Δλ, RMSE. `fit_valid` is a separate uint8 mask for the encoder.

---

## Quick start

```bash
uv sync
```

### Pre-train — bin embedding (RTX 4090)

```bash
uv run python train_pretrain.py \
  --config config/config_libs_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --experiment_name libs_bin_pretrain \
  --num_workers 4
```

### Pre-train — line-token (runtime embedding)

```bash
uv run python train_pretrain.py \
  --config config/config_libs_token_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml \
  --experiment_name libs_line_pretrain \
  --num_workers 4
```

First run builds `line_dict_*.h5` and `line_features_*.h5` (Voigt fits can take hours on full data).

### Pre-train — line-token-linear (pre-baked tokens + `nn.Linear`)

```bash
# Optional: materialize line_tokens_*.h5 first (skipped if already cached)
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml

uv run python train_pretrain.py \
  --config config/config_libs_token_linear_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml \
  --experiment_name libs_token_linear_pretrain \
  --num_workers 4
```

Training only touches `line_tokens_*.h5` (dictionary and Voigt caches remain as intermediate artefacts).

### Fine-tune (binned quantification)

```bash
uv run python train_finetune.py \
  --config config/config_libs_4090.yaml \
  --pretrain_run_dir runs/pretrain_<your_run> \
  --libs_data_config config/libs_data.yaml \
  --task quantification_binned \
  --pool cls_mean \
  --experiment_name libs_binned_ft
```

Use the **same** `--config`, `--libs_data_config`, and (for line-token) `--line_embedding_config` as pre-training.

```bash
uv run python list_runs.py --latest
uv run tensorboard --logdir runs/
```

---

## Configuration

### Model / training

| Config | Embedding | GPU | Notes |
|--------|-----------|-----|--------|
| `config/config_libs_smoke.yaml` | bin | any | Smoke wiring |
| `config/config_libs_4090.yaml` | bin | RTX 4090 | ~17k bins, 5 pretrain epochs |
| `config/config_libs_a100.yaml` | bin | A100 | 60/30 epochs |
| `config/config_libs_token_4090.yaml` | line_token | RTX 4090 | ~8k lines (default threshold) |
| `config/config_libs_token.yaml` | line_token | any | Small model for smoke |
| `config/config_libs_token_linear_4090.yaml` | line_token_linear | RTX 4090 | Pre-baked tokens + `nn.Linear` |
| `config/config_libs_token_linear.yaml` | line_token_linear | any | Smoke for token-linear path |

### Data and line embedding

| Config | Role |
|--------|------|
| `config/libs_data.yaml` | Full dataset (~2256 types × 50 shots) |
| `config/libs_data_smoke.yaml` | 3 types × 6 shots |
| `config/line_embedding.yaml` | Te/Ne grid, `intensity_threshold`, Voigt-fit settings |
| `config/line_embedding_smoke.yaml` | `max_lines: 400` for fast tests |

**Line threshold:** `line_dictionary.intensity_threshold` in `line_embedding.yaml` (default `1e-35`; `1e-33` keeps ~2,425 lines vs ~8,119 at `1e-35`).

---

## Training CLI (essentials)

| Flag | Description |
|------|-------------|
| `--config` | Model size, epochs, batch size |
| `--libs_data_config` | **Required** for real LIBS data (`libs_data.yaml`) |
| `--line_embedding_config` | Line-token modes (`line_embedding.yaml`); defaults to `line_token_linear` unless config sets `embedding_type` |
| `--experiment_name` | Run folder suffix |
| `--num_workers` | DataLoader workers (default 0) |

**Fine-tune:** `--pretrain_run_dir`, `--task` (`quantification_binned`, `classification`, …), `--pool` (`cls`, `mean`, `cls_mean`).

---

## Compare bin vs line-token (same data)

```bash
LIBS=config/libs_data.yaml
SEED=42

# Bin
uv run python train_pretrain.py --config config/config_libs_4090.yaml \
  --libs_data_config $LIBS --seed $SEED --experiment_name compare_bin

# Line-token (runtime embedding)
uv run python train_pretrain.py --config config/config_libs_token_4090.yaml \
  --libs_data_config $LIBS --line_embedding_config config/line_embedding.yaml \
  --seed $SEED --experiment_name compare_line

# Line-token-linear (pre-baked tokens)
uv run python train_pretrain.py --config config/config_libs_token_linear_4090.yaml \
  --libs_data_config $LIBS --line_embedding_config config/line_embedding.yaml \
  --seed $SEED --experiment_name compare_line_linear
```

Match `d_model`, `n_layers`, and epochs in both configs before comparing metrics.

---

## Project layout (main paths)

```
LIBS_foundation/
├── config/
│   ├── config_libs_{4090,a100,smoke}.yaml      # bin embedding
│   ├── config_libs_token_{4090,}.yaml          # line_token (runtime embedding)
│   ├── config_libs_token_linear_{4090,}.yaml   # line_token_linear (pre-baked tokens)
│   ├── libs_data.yaml, libs_data_smoke.yaml
│   └── line_embedding.yaml, line_embedding_smoke.yaml
├── data/
│   ├── libs_pipeline.py
│   ├── line_dictionary.py, line_features.py, line_tokenization.py
│   ├── line_embedding_pipeline.py
│   └── dataset.py
├── scripts/build_line_tokens.py                # offline tokenization CLI
├── models/
│   ├── libs_transformer.py, line_token_embedding.py, heads.py
├── training/pretrain.py, training/finetune.py
├── train_pretrain.py, train_finetune.py
└── run_pretrain_libs.slurm, run_finetune_libs.slurm
```

---

## SLURM (A100, bin embedding)

```bash
sbatch run_pretrain_libs.slurm
sbatch run_finetune_libs.slurm runs/pretrain_<timestamp>_libs_pretrain_a100
```

Extend SLURM scripts with `--line_embedding_config config/line_embedding.yaml` and `config_libs_token_4090.yaml` (runtime) or `config_libs_token_linear_4090.yaml` (pre-baked tokens) for line-token runs.

---

## License

MIT
