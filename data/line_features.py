"""
Per-spectrum Voigt fits at theoretical line centres (precomputed HDF5 cache).

Feature channels per line (index):
  0 max_intensity (amplitude)
  1 FWHM (nm, Olivero–Longbothum)
  2 R²
  3 delta_lambda (fitted centre − theoretical centre)
  4 RMSE
  5 fit_valid (1 = success, 0 = failure)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import wofz

# Feature layout
FEAT_MAX_INT = 0
FEAT_FWHM = 1
FEAT_R2 = 2
FEAT_DELTA_LAM = 3
FEAT_RMSE = 4
FEAT_VALID = 5
N_FEATURES = 6


def _config_hash(parts: dict) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def voigt(x, x0, amplitude, gamma, sigma):
    """Voigt profile; guards against overflow during curve_fit exploration."""
    sigma = max(float(sigma), 1e-6)
    gamma = max(float(gamma), 0.0)
    z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
    with np.errstate(over="ignore", invalid="ignore"):
        profile = wofz(z).real
    denom = sigma * np.sqrt(2 * np.pi)
    out = amplitude * profile / denom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fwhm_voigt(gamma: float, sigma: float) -> float:
    """Olivero & Longbothum (1977) approximation for FWHM in nm."""
    return 0.5346 * (2 * gamma) + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2 * sigma * np.sqrt(2 * np.log(2))) ** 2)


def fit_line_in_spectrum(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    centre_nm: float,
    window_nm: float,
    gamma_init: float,
    sigma_init: float,
    r2_min: float,
) -> np.ndarray:
    """
    Fit a single Voigt to a spectrum window. Returns float32 vector of length N_FEATURES.
    """
    out = np.zeros(N_FEATURES, dtype=np.float32)
    b1_w = centre_nm - window_nm
    b2_w = centre_nm + window_nm
    b1 = int(np.argmin(np.abs(wavelength - b1_w)))
    b2 = int(np.argmin(np.abs(wavelength - b2_w)))
    if b2 <= b1 + 2:
        return out

    x = wavelength[b1:b2]
    y = spectrum[b1:b2].astype(np.float64)
    if y.size < 4 or not np.all(np.isfinite(y)):
        return out

    a1 = int(np.argmax(y))
    x0_guess = float(x[a1])
    y_max = float(np.max(y))
    if y_max <= 0:
        return out
    # Keep the optimizer away from sigma→0 and huge amplitudes (source of overflow warnings).
    lb = [float(x[0]), 0.0, 1e-4, 1e-4]
    ub = [float(x[-1]), y_max * 100.0, 0.5, 0.05]
    try:
        # Covariance is unused; ill-conditioned windows often trigger OptimizeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                voigt, x, y,
                p0=[x0_guess, y_max, gamma_init, sigma_init],
                bounds=(lb, ub),
                maxfev=2000,
            )
        if not np.all(np.isfinite(popt)):
            return out
        x0_fit, amp, gamma, sigma = popt
        if amp <= 0 or sigma <= 0:
            return out
        fit_y = voigt(x, *popt)
        if not np.all(np.isfinite(fit_y)):
            return out
        rmse = float(np.sqrt(np.mean((y - fit_y) ** 2)))
        ss_res = float(np.sum((y - fit_y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else -np.inf
        if not np.isfinite(r2) or r2 < r2_min:
            return out
        out[FEAT_MAX_INT] = float(amp)
        out[FEAT_FWHM] = float(fwhm_voigt(gamma, sigma))
        out[FEAT_R2] = float(r2)
        out[FEAT_DELTA_LAM] = float(x0_fit - centre_nm)
        out[FEAT_RMSE] = rmse
        out[FEAT_VALID] = 1.0
    except (RuntimeError, ValueError, TypeError):
        pass
    return out


# Multiprocessing globals
_w_spectra: np.ndarray | None = None
_w_wavelength: np.ndarray | None = None
_w_centres: np.ndarray | None = None
_w_fit_cfg: dict | None = None


def _init_fit_worker(spectra, wavelength, centres, fit_cfg):
    global _w_spectra, _w_wavelength, _w_centres, _w_fit_cfg
    _w_spectra = spectra
    _w_wavelength = wavelength
    _w_centres = centres
    _w_fit_cfg = fit_cfg


def _fit_spectrum_row(spec_idx: int) -> tuple[int, np.ndarray]:
    n_lines = _w_centres.size
    row = np.zeros((n_lines, N_FEATURES), dtype=np.float32)
    spec = _w_spectra[spec_idx]
    for j in range(n_lines):
        row[j] = fit_line_in_spectrum(
            spec, _w_wavelength, float(_w_centres[j]),
            _w_fit_cfg["window_nm"],
            _w_fit_cfg["gamma_init"],
            _w_fit_cfg["sigma_init"],
            _w_fit_cfg["r2_min"],
        )
    return spec_idx, row


def build_line_features_cache(
    spectra: np.ndarray,
    wavelength: np.ndarray,
    line_dict_path: str,
    fit_cfg: dict,
    spectra_cache_key: str = "",
    verbose: bool = True,
) -> str:
    """
    Build or load [n_spectra, n_lines, 6] feature cache.

    Returns:
        Path to HDF5 file.
    """
    with h5py.File(line_dict_path, "r") as f:
        dict_hash = f.attrs.get("config_hash", "")
        centres = f["central_wavelength"][:]

    cache_dir = Path(fit_cfg.get("cache_dir", "external_data/cache"))
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parents[1] / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    hash_parts = {
        "dict_hash": dict_hash,
        "spectra_key": spectra_cache_key,
        "n_spectra": int(spectra.shape[0]),
        "n_lines": int(centres.size),
        "wavelength_len": int(wavelength.size),
        "fit": {k: v for k, v in fit_cfg.items() if k != "cache_dir" and k != "workers"},
    }
    key = _config_hash(hash_parts)
    out_path = cache_dir / f"line_features_{key}.h5"

    if out_path.is_file():
        if verbose:
            print(f"Line features cache hit: {out_path}")
        return str(out_path)

    n_spec, n_lines = spectra.shape[0], centres.size
    if verbose:
        print(f"Fitting Voigt for {n_spec} spectra × {n_lines} lines → {out_path}")

    workers = int(fit_cfg.get("workers", 1))
    fit_params = {
        "window_nm": float(fit_cfg["window_nm"]),
        "gamma_init": float(fit_cfg["gamma_init"]),
        "sigma_init": float(fit_cfg["sigma_init"]),
        "r2_min": float(fit_cfg["r2_min"]),
    }

    features = np.zeros((n_spec, n_lines, N_FEATURES), dtype=np.float32)

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_fit_worker,
            initargs=(spectra, wavelength, centres, fit_params),
        ) as pool:
            for i, (idx, row) in enumerate(
                pool.imap_unordered(_fit_spectrum_row, range(n_spec), chunksize=max(1, n_spec // (workers * 4)))
            ):
                features[idx] = row
                if verbose and (i + 1) % max(1, n_spec // 20) == 0:
                    print(f"  {i + 1}/{n_spec} spectra")
    else:
        _init_fit_worker(spectra, wavelength, centres, fit_params)
        for idx in range(n_spec):
            _, row = _fit_spectrum_row(idx)
            features[idx] = row
            if verbose and (idx + 1) % max(1, n_spec // 20) == 0:
                print(f"  {idx + 1}/{n_spec} spectra")

    valid_frac = float(features[:, :, FEAT_VALID].mean())
    if verbose:
        print(f"  fit_valid fraction: {valid_frac:.2%}")

    with h5py.File(out_path, "w") as f:
        f.attrs["config_hash"] = key
        f.attrs["line_dict_path"] = line_dict_path
        f.attrs["n_spectra"] = n_spec
        f.attrs["n_lines"] = n_lines
        f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        f.create_dataset("central_wavelength", data=centres)

    if verbose:
        print(f"Saved line features: {out_path}")
    return str(out_path)


class LineFeaturesStore:
    """Lazy HDF5 reader for per-spectrum line features."""

    def __init__(self, path: str):
        self.path = path
        self._file: h5py.File | None = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    @property
    def n_lines(self) -> int:
        self._ensure_open()
        return int(self._file.attrs["n_lines"])

    def get(self, spectrum_idx: int) -> np.ndarray:
        self._ensure_open()
        return self._file["features"][spectrum_idx]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
