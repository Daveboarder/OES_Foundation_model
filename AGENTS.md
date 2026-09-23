# Repository Guidelines

## Project Structure & Module Organization
This Python project trains a self-supervised LIBS spectroscopy transformer. `data/` contains synthesis, datasets, and line-token preprocessing; `models/` contains encoders and prediction heads; `training/` implements Lightning training; `cf/` contains calibration-free physics solvers. Root scripts handle training, evaluation, and figures. Keep experiment YAMLs in `config/`, utility CLIs and sanity checks in `scripts/`, reference inputs in `external_data/`, and curated spectral lines in `cf/data/`. See `README.md` and `ARCHITECTURE.md` before changing pipeline contracts.

## Build, Test, and Development Commands
Run commands from the repository root with Python 3.10+:
- `uv sync --extra dev`: install locked runtime dependencies plus pytest, Black, and Ruff.
- `uv build`: build distributions using Hatchling.
- `uv run python train_pretrain.py --config config/config_libs_smoke.yaml --libs_data_config config/libs_data_smoke.yaml`: exercise pretraining with a small dataset; this config requests a GPU.
- `uv run python scripts/check_libs_pipeline.py`: check plasma ranges, concentration normalization, and generated spectra.
- `uv run python scripts/check_cf_layer.py`: check solver agreement, gradients, and degenerate inputs.
- `uv run python scripts/check_two_zone_physics.py`: validate generator physics and dataset plumbing.
- `uv run black --check .` and `uv run ruff check .`: check formatting and lint.
- `uv run tensorboard --logdir runs/`: inspect training logs.

## Coding Style & Naming Conventions
Use four-space indentation, `snake_case` modules/functions, `PascalCase` classes, and uppercase constants. Black and Ruff target Python 3.10 with 100-character lines. Add type hints and document tensor shapes and physical units. Follow existing YAML naming patterns such as `config_libs_*_smoke.yaml`; keep experiment settings out of implementation code.

## Testing Guidelines
Pytest is available, but no dedicated test suite or coverage threshold is currently committed. Add focused regression tests as `tests/test_<module>.py` and run `uv run pytest` when tests exist. Run relevant sanity scripts for numerical changes and record failures or skipped checks. Use smoke configs before full training.

## Commit & Pull Request Guidelines
History uses imperative subjects such as “Optimize memory usage in spectra handling and multiprocessing,” often followed by explanatory bullets. Keep commits focused. PRs should describe behavior changes, link relevant issues, list validation commands/results, and include configs, seeds, and comparison metrics for model changes. Attach figures when visualization output changes.

## Data & Configuration
Keep checkpoints, HDF5 caches, runs, and secrets out of commits; follow `.gitignore`. Preserve compatible model/data/embedding configs across pretraining and fine-tuning: embedding-mode checkpoints are not interchangeable.
