"""
Two-zone LTE radiative-transfer spectrum generator.

Extends the single-zone physics in SpectraGenerator.py with a simplified 1D
three-segment model (outer → inner → outer) following Zaitsev et al. 2024.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from scipy.special import wofz

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
    window: float = 1.5,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_candidates = [
        _default_db_path(),
        os.path.join(os.path.dirname(__file__), "..", "Source", "LIBS_data_vacuum.db"),
    ]
    db_path = next((p for p in db_candidates if os.path.isfile(p)), db_candidates[0])

    # Table 2 experimental Al-alloy parameters (Zaitsev et al. 2024, Fig. 3)
    te1 = 11510.0
    te2 = 6800.0
    ne1 = 10.0**17.397
    ne2 = 10.0**16.4
    r11_mm = 1.46
    r12_mm = 3.36
    n_density = 1e-4
    l_path = 1.4e-4

    wl_grid = np.linspace(390.0, 400.0, 8000)
    element = "Al"

    one_zone = create_spectra(
        element,
        wl_grid,
        Te=te1,
        Ne=ne1,
        N=n_density,
        l=l_path,
        db_path=db_path,
    )
    two_zone = create_spectra_two_zone_from_radii(
        element,
        wl_grid,
        Te1=te1,
        Ne1=ne1,
        Te2=te2,
        Ne2=ne2,
        r11_mm=r11_mm,
        r12_mm=r12_mm,
        N=n_density,
        db_path=db_path,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(wl_grid, one_zone, label="One zone (Te1, Ne1)", alpha=0.8)
    axes[0].plot(wl_grid, two_zone, label="Two zone (hot core + cold shell)", alpha=0.8)
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Intensity (a.u.)")
    axes[0].set_title(f"{element}: one-zone vs two-zone (390–400 nm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoom on Al I 394.4 nm — self-reversal shows as central dip
    zoom_lo, zoom_hi = 394.0, 395.0
    mask = (wl_grid >= zoom_lo) & (wl_grid <= zoom_hi)
    axes[1].plot(wl_grid[mask], one_zone[mask], label="One zone", alpha=0.8)
    axes[1].plot(wl_grid[mask], two_zone[mask], label="Two zone", alpha=0.8)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].set_title("Zoom: Al I ~394.4 nm (self-reversal)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), "two_zone_demo.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    print(f"Output length: {two_zone.shape[0]} (expected {wl_grid.size})")

    peak_idx = int(np.argmax(two_zone[mask]))
    seg_wl = wl_grid[mask]
    seg_i = two_zone[mask]
    centre_idx = int(np.argmin(np.abs(seg_wl - 394.512)))
    centre_ratio = float(seg_i[centre_idx]) / float(seg_i[peak_idx])
    print(f"Two-zone Al I 394.5 nm centre/wing ratio: {centre_ratio:.3f} (<1 = self-reversal)")
