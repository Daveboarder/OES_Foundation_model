import numpy as np
import os
import scipy.constants as const
from LIBSmethods import voigt, partition_function_cached
import sqlite3
import pandas as pd

# Constants
kb = const.k * 10**7       # erg/K
h = const.h * 10**7        # erg*s
c = const.c                # m/s
e = const.e                # C
me = const.electron_mass * 1000  # g

#print(f"kb: {kb}")
#print(f"h: {h}")
#print(f"c: {c}")
#print(f"e: {e}")
#print(f"me: {me}")

# ---------------------------------------------------------------------------
# Module-level caches (populated lazily, survive the whole process lifetime)
# ---------------------------------------------------------------------------
_db_connection = None
_quant_cache: dict[str, pd.DataFrame] = {}
_eion_cache: dict[str, float] = {}


def _get_connection(db_path: str):
    """Reuse a single read-only SQLite connection across all calls."""
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect(db_path)
    return _db_connection


def _get_quant_param(element: str, db_path: str) -> pd.DataFrame:
    """Fetch QuantParam rows for *element* (cached after first call)."""
    if element not in _quant_cache:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Elem_name, ion_state, Wavelength, Ei, Ek, gi, gk, Ak "
            "FROM QuantParam WHERE Elem_name = ?",
            (element,),
        )
        _quant_cache[element] = pd.DataFrame(
            cursor.fetchall(),
            columns=["Elem_name", "ion_state", "Wavelength", "Ei", "Ek", "gi", "gk", "Ak"],
        )
    return _quant_cache[element]


def _get_eion(element: str, db_path: str) -> float:
    """Fetch ionisation energy for *element* (cached after first call)."""
    if element not in _eion_cache:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Eion FROM E_ion WHERE Elem_name = ?", (element + "+I",)
        )
        result = cursor.fetchall()
        if not result:
            raise ValueError(
                f"Ionization energy (E_ion) not found for element "
                f"'{element}+I' in database."
            )
        _eion_cache[element] = result[0][0]
    return _eion_cache[element]


# ---------------------------------------------------------------------------
# Voigt profile constants (fixed across all calls)
# ---------------------------------------------------------------------------
_GAMMA_FIT = 0.1
_SIGMA_FIT = 0.006
_SIGMA_SQRT2 = _SIGMA_FIT * np.sqrt(2)
_NORM = 1.0 / (_SIGMA_FIT * np.sqrt(2 * np.pi))


def _voigt_all_lines(wavelength: np.ndarray,
                     wavelength_lines: np.ndarray,
                     amplitudes: np.ndarray,
                     window: float = 1.5) -> np.ndarray:
    """
    Windowed Voigt profile: for each spectral line, only evaluate wavelength
    points within *window* nm of the line centre. Much faster than the full
    (W, L) broadcast when the profile is narrow relative to the array span.
    """
    from scipy.special import wofz

    result = np.zeros_like(wavelength)
    n_wl = wavelength.size

    # Pre-sort wavelength once so we can use searchsorted per line
    sorted_idx = np.argsort(wavelength)
    wl_sorted = wavelength[sorted_idx]

    for k in range(wavelength_lines.size):
        centre = wavelength_lines[k]
        amp = amplitudes[k]
        lo = np.searchsorted(wl_sorted, centre - window, side='left')
        hi = np.searchsorted(wl_sorted, centre + window, side='right')
        if lo >= hi:
            continue
        idx = sorted_idx[lo:hi]
        z = (wavelength[idx] - centre + 1j * _GAMMA_FIT) / _SIGMA_SQRT2
        result[idx] += amp * wofz(z).real * _NORM

    return result


def create_spectra(element, wavelength, Te=12705, Ne=1.79e+18,
                   N=1e-4, C=1, l=1.4e-04, db_path=None):
    """
    Generate synthetic optical emission spectra for a given element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Li', 'Cu')
    wavelength : array-like
        Wavelength array (nm)
    Te : float
        Temperature in Kelvin
    Ne : float
        Electron number density in cm^-3
    N : float
        Number density in cm^-3
    C : float
        Content of element (1 = 100 %)
    l : float
        Optical path length in cm
    db_path : str, optional
        Path to LIBS_data_vacuum.db (auto-resolved if None)

    Returns
    -------
    numpy.ndarray
        Synthetic spectrum intensity evaluated at *wavelength*.
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), 'LIBS_data_vacuum.db')

    QuantParam = _get_quant_param(element, db_path)
    if QuantParam.empty:
        return np.zeros_like(wavelength, dtype=float)

    E_ion = _get_eion(element, db_path)

    PF_I, PF_II = partition_function_cached(element, Te, db_path)

    S10 = (
        ((2 * PF_II) / (Ne * PF_I))
        * ((me * kb * Te) / ((h**2) / (2 * np.pi))) ** 1.5
        * np.exp(-(E_ion * 1.60217e-12) / (kb * Te))
    )

    ion_is_I = (QuantParam["ion_state"] == "I").values
    pf_per_line = np.where(ion_is_I, PF_I, PF_II)
    ri = np.where(ion_is_I, 1 / (1 + S10), S10 / (1 + S10))

    wl = QuantParam["Wavelength"].values
    Ak = QuantParam["Ak"].values
    gk = QuantParam["gk"].values
    gi = QuantParam["gi"].values
    Ei = QuantParam["Ei"].values
    Ek = QuantParam["Ek"].values

    eV_to_erg = 1.60217e-12
    kbT = kb * Te

    kt = (
        (wl**4 / (8 * np.pi * c))
        * (Ak * gk * np.exp(-Ei * eV_to_erg / kbT))
        * (1 - np.exp(-eV_to_erg * (Ek - Ei) / kbT))
        / pf_per_line
    )

    Lp = (8 * np.pi * h * c) / (10 * wl**3) * N * np.exp(-eV_to_erg * (Ek - Ei) / kbT) * (gk / gi)

    tau = C * N * ri * l * kt
    Ifin = Lp * (1 - np.exp(-tau))

    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    return _voigt_all_lines(wavelength_arr, wl, Ifin)
