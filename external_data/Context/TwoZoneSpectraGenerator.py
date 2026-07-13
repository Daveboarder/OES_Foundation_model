"""
Two-zone LTE radiative-transfer spectrum generator.

Extends the single-zone physics in SpectraGenerator.py with a simplified 1D
three-segment model (outer → inner → outer) following Zaitsev et al. 2024.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import wofz

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data.libs_pipeline import compute_ccd_wavelengths  # noqa: E402

from LIBSmethods import partition_function_cached
from SpectraGenerator import (
    _GAMMA_FIT,
    _NORM,
    _SIGMA_SQRT2,
    _get_eion,
    _get_quant_param,
    create_spectra,
    c,
    h,
    kb,
    me,
)

# Voigt peak value at line centre (for normalising profile to phi(centre)=1)
_VOIGT_PEAK = wofz(1j * _GAMMA_FIT / _SIGMA_SQRT2).real * _NORM

# Scales mm plasma radii to effective cm path lengths compatible with kappa
# from SpectraGenerator (calibrated at l ~ 1e-4 cm).
_DEFAULT_PATH_SCALE = 2e-3

_EV_TO_ERG = 1.60217e-12


def _default_db_path() -> str:
    return os.path.join(os.path.dirname(__file__), "LIBS_data_vacuum.db")


def _line_source_and_kappa(
    element: str,
    Te: float,
    Ne: float,
    N: float,
    C: float,
    db_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-line source function Lp and absorption coefficient kappa (cm^-1).

    Returns (wavelength_nm, Lp, kappa). Empty arrays if element has no lines.
    """
    quant_param = _get_quant_param(element, db_path)
    if quant_param.empty:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    e_ion = _get_eion(element, db_path)
    pf_i, pf_ii = partition_function_cached(element, Te, db_path)

    s10 = (
        ((2 * pf_ii) / (Ne * pf_i))
        * ((me * kb * Te) / ((h**2) / (2 * np.pi))) ** 1.5
        * np.exp(-(e_ion * _EV_TO_ERG) / (kb * Te))
    )

    ion_is_i = (quant_param["ion_state"] == "I").values
    pf_per_line = np.where(ion_is_i, pf_i, pf_ii)
    ri = np.where(ion_is_i, 1 / (1 + s10), s10 / (1 + s10))

    wl = quant_param["Wavelength"].values.astype(np.float64)
    ak = quant_param["Ak"].values.astype(np.float64)
    gk = quant_param["gk"].values.astype(np.float64)
    gi = quant_param["gi"].values.astype(np.float64)
    ei = quant_param["Ei"].values.astype(np.float64)
    ek = quant_param["Ek"].values.astype(np.float64)
    kbt = kb * Te

    kt = (
        (wl**4 / (8 * np.pi * c))
        * (ak * gk * np.exp(-ei * _EV_TO_ERG / kbt))
        * (1 - np.exp(-_EV_TO_ERG * (ek - ei) / kbt))
        / pf_per_line
    )
    lp = (
        (8 * np.pi * h * c) / (10 * wl**3)
        * N
        * np.exp(-_EV_TO_ERG * (ek - ei) / kbt)
        * (gk / gi)
    )
    kappa = C * N * ri * kt
    return wl, lp, kappa


def _voigt_normalized(wavelength: np.ndarray, centre: float) -> np.ndarray:
    """Voigt line shape normalised to unity at the line centre."""
    z = (wavelength - centre + 1j * _GAMMA_FIT) / _SIGMA_SQRT2
    return wofz(z).real * _NORM / _VOIGT_PEAK


def _radiative_transfer_two_zone(
    s_outer: float | np.ndarray,
    tau_outer: np.ndarray,
    s_inner: float | np.ndarray,
    tau_inner: np.ndarray,
) -> np.ndarray:
    """
    Three-segment recurrence along the observation axis (s=1 base geometry).

    Path: outer (emit) → inner (emit + absorb prior) → outer (emit + absorb prior).

    *tau_outer* and *tau_inner* may be wavelength-dependent arrays; source
    terms *s_outer* / *s_inner* are per-line scalars (Lp at line centre).
    """
    i = s_outer * (1.0 - np.exp(-tau_outer))
    i = i * np.exp(-tau_inner) + s_inner * (1.0 - np.exp(-tau_inner))
    i = i * np.exp(-tau_outer) + s_outer * (1.0 - np.exp(-tau_outer))
    return i


def _two_zone_transfer_all_lines(
    wavelength: np.ndarray,
    line_centres: np.ndarray,
    lp_inner: np.ndarray,
    kappa_inner: np.ndarray,
    lp_outer: np.ndarray,
    kappa_outer: np.ndarray,
    l_inner: float,
    l_outer: float,
    window: float = 15.0,
) -> np.ndarray:
    """
    Sum two-zone radiative transfer for each line with wavelength-dependent tau.

    Self-reversal requires kappa(lambda) to follow the Voigt profile during
    transfer (Zaitsev et al. 2024, Eqs. 1 and 4), not broadening after the fact.
    """
    result = np.zeros_like(wavelength)
    sorted_idx = np.argsort(wavelength)
    wl_sorted = wavelength[sorted_idx]

    for k in range(line_centres.size):
        centre = line_centres[k]
        lo = np.searchsorted(wl_sorted, centre - window, side="left")
        hi = np.searchsorted(wl_sorted, centre + window, side="right")
        if lo >= hi:
            continue
        idx = sorted_idx[lo:hi]
        wl_seg = wavelength[idx]
        phi = _voigt_normalized(wl_seg, centre)
        tau_outer = kappa_outer[k] * phi * l_outer
        tau_inner = kappa_inner[k] * phi * l_inner
        result[idx] += _radiative_transfer_two_zone(
            lp_outer[k], tau_outer, lp_inner[k], tau_inner
        )
    return result


def _path_lengths_from_radii(
    r11_mm: float,
    r12_mm: float,
    path_scale: float = _DEFAULT_PATH_SCALE,
) -> tuple[float, float]:
    """
    Convert semi-ellipse radii (mm) from Zaitsev et al. to cm path segments.

    *path_scale* maps geometric plasma radii to effective optical depths
    compatible with ``kappa`` from SpectraGenerator (calibrated at l ~ 1e-4 cm).
    """
    l_outer = (r12_mm - r11_mm) * 0.1 * path_scale
    l_inner = 2.0 * r11_mm * 0.1 * path_scale
    return l_outer, l_inner


def create_spectra_two_zone(
    element: str,
    wavelength,
    Te1: float,
    Ne1: float,
    Te2: float,
    Ne2: float,
    l_outer: float,
    l_inner: float,
    N: float = 1e-4,
    C: float = 1.0,
    db_path: str | None = None,
) -> np.ndarray:
    """
    Generate a two-zone synthetic spectrum for one element.

    Zone 1 (inner, hot): Te1, Ne1, path length l_inner.
    Zone 2 (outer, cold): Te2, Ne2, path length l_outer (both outer segments).

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Al', 'Cu').
    wavelength : array-like
        Wavelength grid (nm).
    Te1, Ne1 : float
        Inner-zone temperature (K) and electron density (cm^-3).
    Te2, Ne2 : float
        Outer-zone temperature (K) and electron density (cm^-3).
    l_outer, l_inner : float
        Optical path lengths (cm) for outer and inner segments.
    N : float
        Number density (cm^-3).
    C : float
        Element mass fraction / content (1 = 100 %).
    db_path : str, optional
        Path to LIBS_data_vacuum.db.

    Returns
    -------
    numpy.ndarray
        Spectrum on *wavelength* (Voigt profile applied inside radiative transfer).
    """
    if db_path is None:
        db_path = _default_db_path()

    wavelength_arr = np.asarray(wavelength, dtype=np.float64)

    wl_inner, lp_inner, kappa_inner = _line_source_and_kappa(
        element, Te1, Ne1, N, C, db_path
    )
    if wl_inner.size == 0:
        return np.zeros_like(wavelength_arr, dtype=float)

    _, lp_outer, kappa_outer = _line_source_and_kappa(
        element, Te2, Ne2, N, C, db_path
    )

    return _two_zone_transfer_all_lines(
        wavelength_arr,
        wl_inner,
        lp_inner,
        kappa_inner,
        lp_outer,
        kappa_outer,
        l_inner,
        l_outer,
    )


def create_spectra_two_zone_from_radii(
    element: str,
    wavelength,
    Te1: float,
    Ne1: float,
    Te2: float,
    Ne2: float,
    r11_mm: float,
    r12_mm: float,
    N: float = 1e-4,
    C: float = 1.0,
    path_scale: float = _DEFAULT_PATH_SCALE,
    db_path: str | None = None,
) -> np.ndarray:
    """Two-zone spectrum using semi-ellipse radii r11, r12 (mm) from Zaitsev et al."""
    l_outer, l_inner = _path_lengths_from_radii(r11_mm, r12_mm, path_scale=path_scale)
    return create_spectra_two_zone(
        element,
        wavelength,
        Te1,
        Ne1,
        Te2,
        Ne2,
        l_outer,
        l_inner,
        N=N,
        C=C,
        db_path=db_path,
    )


def create_composite_spectrum_two_zone(
    elements: Sequence[str],
    mass_fractions: Sequence[float],
    wavelength,
    Te1: float,
    Ne1: float,
    Te2: float,
    Ne2: float,
    l_outer: float,
    l_inner: float,
    N: float = 1e-4,
    db_path: str | None = None,
) -> np.ndarray:
    """Sum per-element two-zone spectra weighted by mass fraction."""
    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    spectrum = np.zeros_like(wavelength_arr, dtype=float)
    for element, frac in zip(elements, mass_fractions):
        spectrum += create_spectra_two_zone(
            element,
            wavelength_arr,
            Te1,
            Ne1,
            Te2,
            Ne2,
            l_outer,
            l_inner,
            N=N,
            C=float(frac),
            db_path=db_path,
        )
    return spectrum


def create_composite_spectrum_one_zone(
    elements: Sequence[str],
    mass_fractions: Sequence[float],
    wavelength,
    Te: float,
    Ne: float,
    N: float = 1e-4,
    l: float = 1.4e-4,
    db_path: str | None = None,
) -> np.ndarray:
    """Sum per-element single-zone (non-self-reversed) spectra weighted by mass fraction.

    Uses the homogeneous-plasma model from ``SpectraGenerator.create_spectra``.
    Self-absorption is still present via the ``(1 - exp(-tau))`` saturation term,
    but there is no cold outer shell, so strong lines saturate rather than
    developing the central self-reversal dip produced by the two-zone model.
    """
    if db_path is None:
        db_path = _default_db_path()
    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    spectrum = np.zeros_like(wavelength_arr, dtype=float)
    for element, frac in zip(elements, mass_fractions):
        spectrum += create_spectra(
            element,
            wavelength_arr,
            Te=Te,
            Ne=Ne,
            N=N,
            C=float(frac),
            l=l,
            db_path=db_path,
        )
    return spectrum


# ─────────────────────────────────────────────────────────────────────────────
# Resource loaders: VASKUT wavelength axis + Fe-matrix sample compositions
# ─────────────────────────────────────────────────────────────────────────────
def _default_wavelength_json() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "Data", "VASKUT K8.json"
    )


def _default_samples_xlsx() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "Source", "Samples_Fe_matrix.xlsx"
    )


def _ccd_range_spectrum(
    json_path: str, run_id: int, integration_phase: int, ccd_range: int
) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength axis and raw intensity for one CCD range from a VASKUT JSON."""
    analysis = json.load(open(json_path))["analysis"]
    run = next(r for r in analysis["results"] if r["runId"] == run_id)
    phase = run["spectraData"][integration_phase - 1]
    rng = phase["spectra"][ccd_range - 1]
    intensity = np.asarray(rng["results"], dtype=np.float64)
    drift = [rng["drift"]["beta"], rng["drift"]["alpha"]]
    pixel_to_wl = rng["pixelToWaveLength"][::-1]
    wavelength = compute_ccd_wavelengths(intensity.size, pixel_to_wl, drift)
    return wavelength, intensity


def _ccd_range_wavelength(
    json_path: str, run_id: int, integration_phase: int, ccd_range: int
) -> np.ndarray:
    """Per-pixel wavelength axis for one CCD range from a VASKUT analysis JSON.

    Uses the instrument calibration polynomials (drift + pixel→wavelength) without
    replacing the nonlinear pixel mapping with a uniform linspace grid.
    """
    wavelength, _ = _ccd_range_spectrum(json_path, run_id, integration_phase, ccd_range)
    return wavelength


def load_wavelength_vaskut(
    json_path: str | None = None,
    run_id: int = 1,
    integration_phase: int = 1,
) -> np.ndarray:
    """Full LIBS wavelength axis from a VASKUT-style analysis JSON.

    Concatenates the two physically valid CCD ranges (1 and 2), matching the
    convention used by the training pipeline. For VASKUT K8.json this spans
    roughly 146–419 nm.
    """
    if json_path is None:
        json_path = _default_wavelength_json()
    w1 = _ccd_range_wavelength(json_path, run_id, integration_phase, 1)
    w2 = _ccd_range_wavelength(json_path, run_id, integration_phase, 2)
    return np.concatenate([w1, w2])


def load_measured_spectrum_vaskut(
    json_path: str | None = None,
    run_id: int = 1,
    integration_phase: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Measured LIBS spectrum from a VASKUT-style analysis JSON.

    Returns (wavelength, intensity) on the same native-pixel grid as
    :func:`load_wavelength_vaskut` (CCD ranges 1+2 concatenated).
    """
    if json_path is None:
        json_path = _default_wavelength_json()
    w1, i1 = _ccd_range_spectrum(json_path, run_id, integration_phase, 1)
    w2, i2 = _ccd_range_spectrum(json_path, run_id, integration_phase, 2)
    return np.concatenate([w1, w2]), np.concatenate([i1, i2])


def sample_name_from_vaskut_json(json_path: str) -> str | None:
    """Infer the Fe-matrix sample name from VASKUT JSON metadata.

    Combines ``sampleIds`` prompts 'Sample Name' and 'Sample Description'
    (e.g. VASKUT + K8 → ``VASKUT K8``).
    """
    analysis = json.load(open(json_path))["analysis"]
    sample_ids = analysis.get("sampleIds") or {}
    parts: list[str] = []
    for entry in sample_ids.values():
        val = str(entry.get("value", "")).strip()
        if val:
            parts.append(val)
    if not parts:
        return None
    return " ".join(parts)


def normalize_spectrum(
    spectrum: np.ndarray,
    method: str = "max",
    wavelength: np.ndarray | None = None,
    peak_position: float = 205.63,
    peak_window: float = 0.5,
) -> np.ndarray:
    """Scale a spectrum for shape comparison (measured counts vs synthetic a.u.).

    method="max"    : divide by the global peak.
    method="line"   : divide by the local peak near *peak_position* nm (requires
                      *wavelength*); useful to anchor all traces to a common line.
    method="minmax" : rescale to [0, 1].
    """
    spec = np.asarray(spectrum, dtype=np.float64)
    if method == "max":
        peak = float(spec.max())
        return spec / peak if peak > 0 else spec
    if method == "line":
        if wavelength is None:
            raise ValueError("method='line' requires the wavelength array.")
        wl = np.asarray(wavelength, dtype=np.float64)
        window = (wl >= peak_position - peak_window) & (wl <= peak_position + peak_window)
        peak = float(spec[window].max()) if window.any() else 0.0
        return spec / peak if peak > 0 else spec
    if method == "minmax":
        lo, hi = float(spec.min()), float(spec.max())
        return (spec - lo) / (hi - lo) if hi > lo else spec
    raise ValueError(f"Unknown normalization method: {method!r}")


def _db_elements(db_path: str) -> set[str]:
    """Distinct element symbols that actually have line data in the DB."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT Elem_name FROM QuantParam")
        out = set()
        for (name,) in cur.fetchall():
            if name is None:
                continue
            s = str(name).strip()
            if s and "-II" not in s and s not in {"", "n", "r"}:
                out.add(s)
        return out


def load_sample_composition(
    xlsx_path: str | None = None,
    sample_name: str | None = None,
    db_path: str | None = None,
    min_fraction: float = 0.0,
) -> tuple[str, list[str], list[float]]:
    """Read one sample's elemental composition from Samples_Fe_matrix.xlsx.

    Values in the 'Concentrations' sheet are mass percentages; they are returned
    as mass fractions (``% / 100``). Only elements that (a) have a non-zero
    concentration, (b) exceed *min_fraction*, and (c) exist in the line DB are
    kept, so the composition is directly usable as ``C`` weights.

    Parameters
    ----------
    xlsx_path : str, optional
        Path to the Excel matrix (defaults to ../Source/Samples_Fe_matrix.xlsx).
    sample_name : str, optional
        Row to select from the 'Name *' column. If None, the first row is used.
    db_path : str, optional
        Line database used to drop elements without spectral data.
    min_fraction : float
        Discard elements below this mass fraction (default 0 keeps all non-zero).

    Returns
    -------
    (name, elements, mass_fractions)
    """
    if xlsx_path is None:
        xlsx_path = _default_samples_xlsx()
    if db_path is None:
        db_path = _default_db_path()

    conc = pd.read_excel(xlsx_path, sheet_name="Concentrations")
    conc.columns = [str(col).strip() for col in conc.columns]
    name_col = conc.columns[0]

    if sample_name is None:
        row = conc.iloc[0]
    else:
        matches = conc[conc[name_col].astype(str).str.strip() == str(sample_name).strip()]
        if matches.empty:
            raise ValueError(
                f"Sample '{sample_name}' not found in {os.path.basename(xlsx_path)}."
            )
        row = matches.iloc[0]

    resolved_name = str(row[name_col]).strip()
    db_elems = _db_elements(db_path)

    elements: list[str] = []
    fractions: list[float] = []
    for col in conc.columns[1:]:
        if not col or col.lower().startswith("unnamed:") or col not in db_elems:
            continue
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        frac = float(val) / 100.0
        if frac > min_fraction:
            elements.append(col)
            fractions.append(frac)

    return resolved_name, elements, fractions


def _pick_self_reversal_line(
    wavelength: np.ndarray,
    one_zone: np.ndarray,
    two_zone: np.ndarray,
    n_candidates: int = 60,
    core_nm: float = 0.03,
    flank_nm: float = 0.15,
) -> float:
    """Return the wavelength of the strong line with the clearest two-zone
    self-reversal — a genuine central dip flanked by two humps of the *same*
    line, rather than a neighbouring stronger line.

    A line reverses when the two-zone profile has, on both sides of the line
    core (within ``flank_nm``), a local maximum higher than the core value
    (measured within ``core_nm``). The line with the deepest such dip is
    returned; if none reverse, the strongest single-zone line is used.
    """
    order = np.argsort(one_zone)[::-1]
    seen: list[float] = []
    best_wl = float(wavelength[order[0]])
    best_depth = 1.0
    for idx in order:
        wl0 = float(wavelength[idx])
        if any(abs(wl0 - s) < 2 * flank_nm for s in seen):
            continue
        seen.append(wl0)
        if len(seen) > n_candidates:
            break

        win = (wavelength >= wl0 - flank_nm) & (wavelength <= wl0 + flank_nm)
        # Only consider genuine emission lines (a local peak in the one-zone
        # spectrum), never inter-line valleys.
        if one_zone[idx] < one_zone[win].max() * 0.999:
            continue

        left = two_zone[(wavelength >= wl0 - flank_nm) & (wavelength < wl0 - core_nm)]
        right = two_zone[(wavelength > wl0 + core_nm) & (wavelength <= wl0 + flank_nm)]
        if left.size == 0 or right.size == 0:
            continue

        core_val = float(two_zone[idx])  # intensity at the line centre
        if core_val <= 0:
            continue
        # Reversal: both wings of the SAME line rise above the depressed core.
        flank = min(float(left.max()), float(right.max()))
        depth = flank / core_val  # >1 => central self-reversal dip
        if depth > best_depth:
            best_depth, best_wl = depth, wl0
    return best_wl


def _resolve_db_path() -> str:
    candidates = [
        _default_db_path(),
        os.path.join(os.path.dirname(__file__), "..", "Source", "LIBS_data_vacuum.db"),
    ]
    return next((p for p in candidates if os.path.isfile(p)), candidates[0])


def _db_lines_in_window(
    db_path: str,
    elements: Sequence[str],
    wl_lo: float,
    wl_hi: float,
    top_per_element: int = 3,
) -> list[tuple[str, float, str]]:
    """Strongest DB transition wavelengths per element inside a window."""
    lines: list[tuple[str, float, str]] = []
    with sqlite3.connect(db_path) as conn:
        for elem in elements:
            cur = conn.cursor()
            cur.execute(
                "SELECT Wavelength, ion_state, Ak FROM QuantParam "
                "WHERE Elem_name = ? AND Wavelength BETWEEN ? AND ? "
                "ORDER BY Ak DESC LIMIT ?",
                (elem, wl_lo, wl_hi, top_per_element),
            )
            for wl, ion, _ak in cur.fetchall():
                lines.append((elem, float(wl), str(ion)))
    return lines


def verify_wavelength_alignment(
    wavelength: np.ndarray,
    measured: np.ndarray,
    synthetic: np.ndarray,
    db_path: str,
    ref_line_nm: float = 231.903,
    band: tuple[float, float] = (230.0, 280.0),
) -> None:
    """Print checks that measured peaks align with DB lines on the true pixel axis."""
    idx = int(np.argmin(np.abs(wavelength - ref_line_nm)))
    axis_wl = float(wavelength[idx])
    print(f"Alignment check: DB Fe II {ref_line_nm:.3f} nm -> axis[{idx}] = "
          f"{axis_wl:.3f} nm (delta {axis_wl - ref_line_nm:+.4f} nm)")

    lo, hi = band
    mask = (wavelength >= lo) & (wavelength <= hi)
    seg_wl, seg_meas, seg_syn = wavelength[mask], measured[mask], synthetic[mask]

    def _top_peaks(w: np.ndarray, y: np.ndarray, n: int = 5) -> list[float]:
        peaks: list[tuple[float, float]] = []
        for i in range(2, len(y) - 2):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.3 * y.max():
                peaks.append((float(w[i]), float(y[i])))
        peaks.sort(key=lambda p: -p[1])
        seen: list[float] = []
        out: list[float] = []
        for wl0, _ in peaks:
            if any(abs(wl0 - s) < 0.3 for s in seen):
                continue
            seen.append(wl0)
            out.append(wl0)
            if len(out) >= n:
                break
        return out

    meas_peaks = _top_peaks(seg_wl, seg_meas)
    syn_peaks = _top_peaks(seg_wl, seg_syn, n=30)
    print(f"Peak offsets in {lo:.0f}-{hi:.0f} nm (measured -> nearest synthetic):")
    for mw in meas_peaks:
        if syn_peaks:
            nearest = min(syn_peaks, key=lambda p: abs(p - mw))
            print(f"  {mw:.2f} nm -> {nearest:.2f} nm  (delta {mw - nearest:+.2f} nm)")
        else:
            print(f"  {mw:.2f} nm -> (no synthetic peaks)")


def main() -> None:
    """Generate synthetic spectra for a Fe-matrix composition and compare them
    with the measured spectrum from a VASKUT analysis JSON in Plotly."""
    import argparse

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        default=None,
        help="Sample name from Samples_Fe_matrix.xlsx 'Name *' column. "
        "If omitted, inferred from the VASKUT JSON (e.g. VASKUT K8).",
    )
    parser.add_argument("--json", default=_default_wavelength_json(),
                        help="VASKUT analysis JSON (wavelength + measured spectrum).")
    parser.add_argument("--xlsx", default=_default_samples_xlsx(),
                        help="Fe-matrix sample composition workbook.")
    parser.add_argument("--run-id", type=int, default=1,
                        help="Run id inside the VASKUT JSON (default 1).")
    parser.add_argument("--integration-phase", type=int, default=1,
                        help="Integration phase / laser shot index (default 1).")
    parser.add_argument("--min-fraction", type=float, default=1e-4,
                        help="Drop elements below this mass fraction "
                        "(default 1e-4 = 0.01 %).")
    parser.add_argument(
        "--normalize", choices=("max", "line", "minmax"), default="max",
        help="Per-trace normalization so measured counts and synthetic a.u. "
        "can be compared by shape. 'max'=divide by global peak, "
        "'line'=divide by local peak near 404.69 nm, 'minmax'=rescale to [0,1] "
        "(default: max).",
    )
    parser.add_argument("--out", default=os.path.join(here, "two_zone_sample_comparison.html"),
                        help="Output HTML path for the interactive figure.")
    args = parser.parse_args()

    db_path = _resolve_db_path()
    sample_name = args.sample or sample_name_from_vaskut_json(args.json)
    if sample_name is None:
        raise SystemExit(
            "Could not infer sample name from JSON — pass --sample explicitly."
        )

    # 2 two-zone plasma parameters.
    te1, te2 = 18370.0, 2200.0
    ne1, ne2 = 10.0**17.397, 10.0**16.4
    r11_mm, r12_mm = 1.46, 4.86
    n_density = 1e-4
    l_path = 1.4e-4
    l_outer, l_inner = _path_lengths_from_radii(r11_mm, r12_mm)

    wavelength, measured = load_measured_spectrum_vaskut(
        args.json, run_id=args.run_id, integration_phase=args.integration_phase,
    )
    name, elements, fractions = load_sample_composition(
        args.xlsx, sample_name, db_path=db_path, min_fraction=args.min_fraction,
    )
    print(f"Sample: {name}")
    print(f"VASKUT run {args.run_id}, integration phase {args.integration_phase}")
    print(f"Wavelength axis: {wavelength.size} points, "
          f"{wavelength.min():.1f}-{wavelength.max():.1f} nm")
    print(f"Measured intensity: {measured.min():.0f} – {measured.max():.0f} counts")
    print(f"Elements ({len(elements)}): "
          + ", ".join(f"{e}={f*100:.3g}%" for e, f in zip(elements, fractions)))

    print("Generating non-self-absorbed (single-zone) spectrum...")
    non_self_absorbed = create_composite_spectrum_one_zone(
        elements, fractions, wavelength,
        Te=te1, Ne=ne1, N=n_density, l=l_path, db_path=db_path,
    )
    print("Generating self-absorbed (two-zone) spectrum...")
    self_absorbed = create_composite_spectrum_two_zone(
        elements, fractions, wavelength,
        Te1=te1, Ne1=ne1, Te2=te2, Ne2=ne2,
        l_outer=l_outer, l_inner=l_inner, N=n_density, db_path=db_path,
    )

    measured_n = normalize_spectrum(measured, args.normalize, wavelength=wavelength)
    one_zone_n = normalize_spectrum(non_self_absorbed, args.normalize, wavelength=wavelength)
    two_zone_n = normalize_spectrum(self_absorbed, args.normalize, wavelength=wavelength)

    verify_wavelength_alignment(
        wavelength, measured, self_absorbed, db_path,
    )
    print(
        "Note: delete external_data/cache/synthetic_cache_*.h5 after this "
        "wavelength fix so training regenerates spectra on the true pixel axis."
    )

    # Zoom on the strongest measured emission line.
    peak_idx = int(np.argmax(measured))
    peak_wl = float(wavelength[peak_idx])
    zoom_lo, zoom_hi = peak_wl - 1.0, peak_wl + 1.0
    zoom_mask = (wavelength >= zoom_lo) & (wavelength <= zoom_hi)
    db_lines = _db_lines_in_window(
        db_path, elements[:4], zoom_lo, zoom_hi, top_per_element=2,
    )

    norm_label = {
        "max": "peak-normalized",
        "line": "line-normalized @404.69 nm",
        "minmax": "min–max normalized",
    }.get(args.normalize, args.normalize)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{name}: measured vs synthetic ({norm_label}, "
            f"{wavelength.min():.0f}-{wavelength.max():.0f} nm)",
            f"Zoom on strongest measured line ~{peak_wl:.2f} nm",
        ),
        vertical_spacing=0.12,
    )
    colours = {"meas": "#2ca02c", "non": "#1f77b4", "self": "#d62728"}

    for y, label, colour, row in (
        (measured_n, "Measured (VASKUT)", colours["meas"], 1),
        (one_zone_n, "Synthetic — one zone", colours["non"], 1),
        (two_zone_n, "Synthetic — two zone", colours["self"], 1),
    ):
        fig.add_trace(go.Scatter(
            x=wavelength, y=y, name=label,
            line=dict(color=colour, width=1),
        ), row=row, col=1)

    for y, colour, group in (
        (measured_n, colours["meas"], "meas"),
        (one_zone_n, colours["non"], "non"),
        (two_zone_n, colours["self"], "self"),
    ):
        fig.add_trace(go.Scatter(
            x=wavelength[zoom_mask], y=y[zoom_mask],
            legendgroup=group, showlegend=False,
            line=dict(color=colour, width=2),
        ), row=2, col=1)

    for elem, wl_db, ion in db_lines:
        fig.add_vline(
            x=wl_db, line_width=1, line_dash="dot", line_color="gray", opacity=0.6,
            annotation_text=f"{elem} {ion}", annotation_position="top",
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    y_title = "Normalized intensity" if args.normalize else "Intensity (a.u.)"
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=2, col=1)
    fig.update_layout(
        title=f"Measured vs synthetic LIBS — {name} (VASKUT K8)",
        template="plotly_white", hovermode="x unified", height=850,
    )

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Saved interactive comparison to {args.out}")

    rev_wl = _pick_self_reversal_line(wavelength, non_self_absorbed, self_absorbed)
    core_i = int(np.argmin(np.abs(wavelength - rev_wl)))
    left = self_absorbed[(wavelength >= rev_wl - 0.15) & (wavelength < rev_wl - 0.03)]
    right = self_absorbed[(wavelength > rev_wl + 0.03) & (wavelength <= rev_wl + 0.15)]
    if left.size and right.size:
        centre = float(self_absorbed[core_i])
        wing = min(float(left.max()), float(right.max()))
        ratio = centre / wing if wing else float("nan")
        note = "self-reversal dip" if ratio < 1 else "no reversal"
        print(f"Strongest two-zone reversal at {rev_wl:.2f} nm: "
              f"centre/wing = {ratio:.3f} ({note})")


if __name__ == "__main__":
    main()
