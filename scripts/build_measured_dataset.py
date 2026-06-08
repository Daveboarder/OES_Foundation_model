"""
Build measured LIBS HDF5 cache from Chameleon OptiCal JSON files.

Optionally materializes line-token caches for line_token_linear training.

Usage:
    uv run python scripts/build_measured_dataset.py \\
        --libs_data_config config/libs_data_measured.yaml

    uv run python scripts/build_measured_dataset.py \\
        --libs_data_config config/libs_data_measured.yaml \\
        --line_embedding_config config/line_embedding.yaml \\
        --build_line_tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.libs_pipeline import build_dataset_from_config
from data.line_embedding_pipeline import prepare_line_tokens_assets
from data.measured_pipeline import load_spectrum_from_json, find_first_valid_run
from external_data.Context.readData import json_from_file


def _validate_h5_schema(cache_path: str, expected_n_bins: int) -> None:
    """Assert measured cache matches synthetic cache layout."""
    with h5py.File(cache_path, "r") as f:
        assert "spectra" in f and "sample_table" in f
        spectra = f["spectra"]
        assert spectra.ndim == 2
        assert spectra.shape[1] == expected_n_bins
        grp = f["sample_table"]
        cols = json.loads(grp.attrs["columns"])
        for required in ("sample_type_id", "sample_type_name", "unique_id", "Te", "Ne"):
            assert required in cols, f"missing column {required}"


def _validate_concentrations(table) -> None:
    meta = {"sample_type_id", "sample_type_name", "unique_id", "Te", "Ne"}
    elem_cols = [c for c in table.columns if c not in meta]
    sums = table[elem_cols].sum(axis=1).values
    if not np.allclose(sums, 1.0, atol=1e-4):
        bad = np.where(np.abs(sums - 1.0) > 1e-4)[0]
        raise AssertionError(
            f"concentration rows do not sum to 1.0 (first bad idx: {bad[:5].tolist()})"
        )


def _smoke_test_spectrum(chameleon_root: Path, ref_wavelength: np.ndarray) -> None:
    """Known file from plan: REMUS-9951602/FEGLFE/BAS NIRM1.json, run 2."""
    sample = chameleon_root / "REMUS-9951602" / "FEGLFE" / "BAS NIRM1.json"
    if not sample.is_file():
        print(f"  Smoke test skipped (file not found): {sample}")
        return
    analysis = json_from_file(str(sample))["analysis"]
    run_id = find_first_valid_run(analysis)
    assert run_id is not None, "expected a valid run in smoke-test file"
    spec = load_spectrum_from_json(
        sample, run_id=run_id, reference_wavelength=ref_wavelength,
    )
    assert spec.shape == (len(ref_wavelength),), (
        f"smoke spectrum shape {spec.shape} != ({len(ref_wavelength)},)"
    )
    print(f"  Smoke test OK: {sample.name} run={run_id} shape={spec.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build measured LIBS HDF5 cache from Chameleon JSON files.",
    )
    parser.add_argument("--libs_data_config", type=str, required=True)
    parser.add_argument("--line_embedding_config", type=str, default=None)
    parser.add_argument(
        "--build_line_tokens",
        action="store_true",
        help="Also build line_dict / line_features / line_tokens HDF5 caches",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Override measured.max_files (for smoke tests)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.libs_data_config))
    if args.max_files is not None:
        cfg.setdefault("measured", {})["max_files"] = args.max_files

    paths = cfg["paths"]
    chameleon_root = Path(paths["measured_json_root"]).expanduser().resolve()

    from data.libs_pipeline import load_wavelength
    ref_wl = load_wavelength(str(Path(paths["wavelength_json"]).expanduser().resolve()))
    print("Smoke test...")
    _smoke_test_spectrum(chameleon_root, ref_wl)

    print(f"\nBuilding measured dataset from {args.libs_data_config}...")
    ds = build_dataset_from_config(cfg)
    if len(ds) == 0:
        raise SystemExit("Measured pipeline produced no spectra — check paths and matrix matching.")

    cache_path = Path(ds.cache_dir) / f"measured_cache_{ds.cache_key}.h5"

    print(f"\nDataset: {len(ds)} spectra × {ds.spectra.shape[1]} bins")
    print(f"Cache key: {ds.cache_key}")
    print(f"Cache path: {cache_path}")

    _validate_h5_schema(str(cache_path), expected_n_bins=ds.spectra.shape[1])
    _validate_concentrations(ds.sample_table)
    print("HDF5 schema and concentration checks passed.")

    unique_types = ds.sample_table["sample_type_id"].nunique()
    print(f"Unique sample types: {unique_types}")

    if args.build_line_tokens:
        if not args.line_embedding_config:
            raise SystemExit("--build_line_tokens requires --line_embedding_config")
        print("\nBuilding line-token caches...")
        meta = prepare_line_tokens_assets(
            ds.spectra.astype(np.float32),
            ds.wavelength,
            args.line_embedding_config,
            spectra_cache_key=ds.cache_key,
            verbose=True,
        )
        print(f"  Tokens: {meta['line_tokens_path']}")
        print(f"  Shape:  [{ds.spectra.shape[0]}, {meta['n_lines']}, {meta['n_features']}]")

    print("\nDone. Fine-tune with:")
    print(
        f"  uv run python train_finetune.py \\\n"
        f"      --libs_data_config {args.libs_data_config} \\\n"
        f"      --pretrain_run_dir runs/pretrain_<your_run> \\\n"
        f"      --task quantification_binned"
    )


if __name__ == "__main__":
    main()
