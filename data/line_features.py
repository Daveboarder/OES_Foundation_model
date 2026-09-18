"""
Per-spectrum Voigt fits at theoretical line centres (precomputed HDF5 cache).

Feature channels per line (index):
  0 max_intensity (amplitude = fitted Voigt AREA; the profile is area-normalised)
  1 FWHM (nm, Olivero–Longbothum)
  2 R²
  3 delta_lambda (fitted centre − theoretical centre)
  4 RMSE
  5 fit_valid (1 = success, 0 = failure)

Optional linear baseline (``line_features.baseline: linear``): the model becomes
``voigt(x) + b0 + b1·(x − centre)``.  The six output channels are unchanged
(area stays channel 0); R² is evaluated on the baseline-subtracted data so a
featureless sloped window is not accepted as a valid line, and RMSE is the
full-model residual.  ``baseline: none`` (default) is bit-identical to the
legacy fit and keeps existing cache hashes.
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

BASELINE_MODES = ("none", "linear")


def _config_hash(parts: dict) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _normalise_baseline(value: Any) -> str:
    mode = "none" if value is None else str(value).strip().lower()
    if mode not in BASELINE_MODES:
        raise ValueError(f"line_features.baseline must be one of {BASELINE_MODES}, got {value!r}")
    return mode


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


def _baseline_init_and_bounds(
    x: np.ndarray, y: np.ndarray, centre_nm: float,
) -> tuple[list[float], list[float], list[float]]:
    """Initial guess and bounds for the linear baseline ``b0 + b1·(x − centre)``.

    The offset starts at the baseline level implied by the window edges and
    may range over the data span (down to one span below the minimum); the
    slope is bounded by four data spans over the window width.
    """
    n_edge = max(2, x.size // 8)
    xl, xr = float(np.mean(x[:n_edge])), float(np.mean(x[-n_edge:]))
    yl, yr = float(np.mean(y[:n_edge])), float(np.mean(y[-n_edge:]))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    span = max(y_max - y_min, 1e-9)
    width = max(float(x[-1] - x[0]), 1e-9)
    slope_max = 4.0 * span / width
    b1_0 = (yr - yl) / max(xr - xl, 1e-9)
    b0_0 = 0.5 * (yl + yr) + b1_0 * (centre_nm - 0.5 * (xl + xr))
    lb = [y_min - span, -slope_max]
    ub = [y_max + 1e-9, slope_max]
    eps = 1e-6 * span
    p0 = [
        float(np.clip(b0_0, lb[0] + eps, ub[0] - eps)),
        float(np.clip(b1_0, lb[1] + 1e-6 * slope_max, ub[1] - 1e-6 * slope_max)),
    ]
    return p0, lb, ub


def fit_line_in_spectrum(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    centre_nm: float,
    window_nm: float,
    gamma_init: float,
    sigma_init: float,
    r2_min: float,
    baseline: str = "none",
    max_delta_nm: float | None = None,
    max_fwhm_nm: float | None = None,
) -> np.ndarray:
    """
    Fit a single Voigt (optionally on a linear baseline) to a spectrum window.
    Returns float32 vector of length N_FEATURES; channel 0 is the Voigt area.

    ``max_delta_nm`` bounds the fitted centre to ``centre_nm ± max_delta_nm``
    (instead of the whole window) and ``max_fwhm_nm`` rejects fits broader
    than that; both default to ``None`` (legacy behaviour).  They stop a
    featureless or sloped window from being "explained" by a very broad Voigt
    parked at the window edge, which the baseline term makes more likely.
    """
    baseline = _normalise_baseline(baseline)
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

    x0_lo, x0_hi = float(x[0]), float(x[-1])
    if max_delta_nm is not None:
        x0_lo = max(x0_lo, centre_nm - float(max_delta_nm))
        x0_hi = min(x0_hi, centre_nm + float(max_delta_nm))
        if x0_hi <= x0_lo:
            return out
        in_range = (x >= x0_lo) & (x <= x0_hi)
        if not in_range.any():
            return out
        a1 = int(np.flatnonzero(in_range)[np.argmax(y[in_range])])
    else:
        a1 = int(np.argmax(y))
    x0_guess = float(np.clip(x[a1], x0_lo, x0_hi))
    y_max = float(np.max(y))
    if y_max <= 0:
        return out
    # Keep the optimizer away from sigma→0 and huge amplitudes (source of overflow warnings).
    lb = [x0_lo, 0.0, 1e-4, 1e-4]
    ub = [x0_hi, y_max * 100.0, 0.5, 0.05]
    p0 = [x0_guess, y_max, float(gamma_init), float(sigma_init)]

    if baseline == "linear":
        p0_b, lb_b, ub_b = _baseline_init_and_bounds(x, y, centre_nm)
        p0 = p0 + p0_b
        lb = lb + lb_b
        ub = ub + ub_b
        # The Voigt amplitude guess is the peak above the initial baseline.
        p0[1] = max(y_max - (p0_b[0] + p0_b[1] * (x0_guess - centre_nm)), 1e-6 * y_max)

        def model(xx, x0, amplitude, gamma, sigma, b0, b1_):
            return voigt(xx, x0, amplitude, gamma, sigma) + b0 + b1_ * (xx - centre_nm)
    else:
        model = voigt

    try:
        # Covariance is unused; ill-conditioned windows often trigger OptimizeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                model, x, y,
                p0=p0,
                bounds=(lb, ub),
                maxfev=2000,
            )
        if not np.all(np.isfinite(popt)):
            return out
        x0_fit, amp, gamma, sigma = popt[:4]
        if amp <= 0 or sigma <= 0:
            return out
        fit_y = model(x, *popt)
        if not np.all(np.isfinite(fit_y)):
            return out
        rmse = float(np.sqrt(np.mean((y - fit_y) ** 2)))
        ss_res = float(np.sum((y - fit_y) ** 2))
        # R² of the line against the baseline-subtracted data (identical to the
        # plain R² when there is no baseline): a sloped, featureless window
        # must not pass as a valid line just because the baseline fits.
        if baseline == "linear":
            y_line = y - (popt[4] + popt[5] * (x - centre_nm))
        else:
            y_line = y
        ss_tot = float(np.sum((y_line - np.mean(y_line)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else -np.inf
        if not np.isfinite(r2) or r2 < r2_min:
            return out
        fwhm = float(fwhm_voigt(gamma, sigma))
        if max_fwhm_nm is not None and fwhm > float(max_fwhm_nm):
            return out
        out[FEAT_MAX_INT] = float(amp)
        out[FEAT_FWHM] = fwhm
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
            baseline=_w_fit_cfg.get("baseline", "none"),
            max_delta_nm=_w_fit_cfg.get("max_delta_nm"),
            max_fwhm_nm=_w_fit_cfg.get("max_fwhm_nm"),
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

    ``fit_cfg`` keys: ``window_nm``, ``gamma_init``, ``sigma_init``, ``r2_min``
    (required), ``baseline`` (``none`` | ``linear``, default ``none``),
    ``max_delta_nm`` / ``max_fwhm_nm`` (optional validity limits, default
    ``null`` = legacy), ``workers`` and ``cache_dir`` (not part of the hash).

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

    baseline = _normalise_baseline(fit_cfg.get("baseline", "none"))
    max_delta_nm = fit_cfg.get("max_delta_nm")
    max_fwhm_nm = fit_cfg.get("max_fwhm_nm")
    max_delta_nm = None if max_delta_nm is None else float(max_delta_nm)
    max_fwhm_nm = None if max_fwhm_nm is None else float(max_fwhm_nm)
    # The hash covers every fit key; the legacy defaults (``baseline: none``,
    # unset limits) are dropped so that configs written before these options
    # existed keep their cache files.
    _optional = ("baseline", "max_delta_nm", "max_fwhm_nm")
    fit_hash_cfg = {k: v for k, v in fit_cfg.items() if k not in ("cache_dir", "workers", *_optional)}
    if baseline != "none":
        fit_hash_cfg["baseline"] = baseline
    if max_delta_nm is not None:
        fit_hash_cfg["max_delta_nm"] = max_delta_nm
    if max_fwhm_nm is not None:
        fit_hash_cfg["max_fwhm_nm"] = max_fwhm_nm
    hash_parts = {
        "dict_hash": dict_hash,
        "spectra_key": spectra_cache_key,
        "n_spectra": int(spectra.shape[0]),
        "n_lines": int(centres.size),
        "wavelength_len": int(wavelength.size),
        "fit": fit_hash_cfg,
    }
    key = _config_hash(hash_parts)
    out_path = cache_dir / f"line_features_{key}.h5"

    if out_path.is_file():
        if verbose:
            print(f"Line features cache hit: {out_path}")
        return str(out_path)

    n_spec, n_lines = spectra.shape[0], centres.size
    if verbose:
        print(f"Fitting Voigt for {n_spec} spectra × {n_lines} lines "
              f"(baseline={baseline}) → {out_path}")

    workers = int(fit_cfg.get("workers", 1))
    fit_params = {
        "window_nm": float(fit_cfg["window_nm"]),
        "gamma_init": float(fit_cfg["gamma_init"]),
        "sigma_init": float(fit_cfg["sigma_init"]),
        "r2_min": float(fit_cfg["r2_min"]),
        "baseline": baseline,
        "max_delta_nm": max_delta_nm,
        "max_fwhm_nm": max_fwhm_nm,
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
        f.attrs["baseline"] = baseline
        f.attrs["fit_json"] = json.dumps(fit_params, sort_keys=True)
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
