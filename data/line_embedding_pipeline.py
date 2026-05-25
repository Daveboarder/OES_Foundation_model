"""
Orchestration: build the three HDF5 caches required by the line-token path.

    1. line_dict_<hash>.h5     — theoretical dictionary
    2. line_features_<hash>.h5 — per-spectrum Voigt fits
    3. line_tokens_<hash>.h5   — combined per-spectrum token tensor,
                                 directly consumable by nn.Linear

For the new ``line_token_linear`` model path, only step 3 is touched at
training time (steps 1 and 2 are intermediate artefacts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from data.line_dictionary import build_line_dictionary, load_line_dictionary_meta
from data.line_features import (
    FEAT_FWHM,
    FEAT_MAX_INT,
    FEAT_VALID,
    build_line_features_cache,
)
from data.line_tokenization import build_line_tokens_cache


def load_line_embedding_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_fit_normalization(features_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of fit channels over valid fits only."""
    with h5py.File(features_path, "r") as f:
        feats = f["features"][:]
    valid = feats[:, :, FEAT_VALID] > 0.5
    fit = feats[:, :, :5]
    if not valid.any():
        return np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)
    vals = fit[valid]
    mean = vals.mean(axis=0).astype(np.float32)
    std = vals.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def prepare_line_token_assets(
    spectra: np.ndarray,
    wavelength: np.ndarray,
    line_embedding_config_path: str,
    spectra_cache_key: str = "",
    project_root: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build or load line dictionary and per-spectrum feature HDF5 caches.

    Returns dict with paths, meta for model init, and fit normalization stats.
    """
    project_root = project_root or Path(__file__).resolve().parents[1]
    cfg = load_line_embedding_config(line_embedding_config_path)

    dict_path = build_line_dictionary(cfg, project_root=project_root, verbose=verbose)
    meta = load_line_dictionary_meta(dict_path)

    fit_path = build_line_features_cache(
        spectra=spectra.astype(np.float32),
        wavelength=wavelength,
        line_dict_path=dict_path,
        fit_cfg=cfg["line_features"],
        spectra_cache_key=spectra_cache_key,
        verbose=verbose,
    )
    fit_mean, fit_std = compute_fit_normalization(fit_path)
    meta["fit_mean"] = fit_mean
    meta["fit_std"] = fit_std
    meta["line_dict_path"] = dict_path
    meta["line_features_path"] = fit_path

    return meta


def prepare_line_tokens_assets(
    spectra: np.ndarray,
    wavelength: np.ndarray,
    line_embedding_config_path: str,
    spectra_cache_key: str = "",
    project_root: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build (or reuse) all three caches and return everything the trainer needs
    to wire the ``line_token_linear`` model.

    Returns dict with keys:
        line_dict_path, line_features_path, line_tokens_path,
        n_lines, n_features,
        feature_names, feature_mean (n_features,), feature_std (n_features,),
        central_wavelength (n_lines,)
    """
    project_root = project_root or Path(__file__).resolve().parents[1]
    cfg = load_line_embedding_config(line_embedding_config_path)

    dict_path = build_line_dictionary(cfg, project_root=project_root, verbose=verbose)
    fit_path = build_line_features_cache(
        spectra=spectra.astype(np.float32),
        wavelength=wavelength,
        line_dict_path=dict_path,
        fit_cfg=cfg["line_features"],
        spectra_cache_key=spectra_cache_key,
        verbose=verbose,
    )
    tokens_path = build_line_tokens_cache(
        line_dict_path=dict_path,
        line_features_path=fit_path,
        cache_dir=cfg.get("line_tokens", {}).get("cache_dir"),
        verbose=verbose,
    )

    with h5py.File(tokens_path, "r") as f:
        feature_names = __import__("json").loads(f.attrs["feature_names"])
        feature_mean = np.asarray(f.attrs["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(f.attrs["feature_std"], dtype=np.float32)
        central_wavelength = f["central_wavelength"][:].astype(np.float32)
        n_lines = int(f.attrs["n_lines"])
        n_features = int(f.attrs["n_features"])

    return {
        "line_dict_path": dict_path,
        "line_features_path": fit_path,
        "line_tokens_path": tokens_path,
        "n_lines": n_lines,
        "n_features": n_features,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "central_wavelength": central_wavelength,
    }
