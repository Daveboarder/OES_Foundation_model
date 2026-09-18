"""
Atomic reference data: standard atomic weights and mass/number-fraction conversion.

The sample matrix (``Samples_Fe_matrix.xlsx``) and the certified reference
values are mass fractions (wt %).  Plasma emission physics (Boltzmann /
Saha, closure equation) works with *number* fractions.  Nothing in the repo
converted between the two before this module existed; the legacy generator in
``data/libs_pipeline.py`` feeds mass fractions straight into the intensity
formula.  Use :func:`mass_to_number_fractions` before synthesis and
:func:`number_to_mass_fractions` after a calibration-free closure.

Atomic weights: IUPAC 2013 conventional / abridged standard atomic weights
(u).  Elements without a stable isotope carry the mass number of the
longest-lived isotope.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ATOMIC_MASS_AMU: dict[str, float] = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62,
    "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327,
    "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
    "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422,
    "Yb": 173.045, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0,
    "Ra": 226.0, "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588, "U": 238.02891,
}

AMU_G = 1.66053906660e-24  # gram per atomic mass unit


def atomic_number(symbol: str) -> int:
    """Atomic number Z. Delegates to the tokenizer's table (single source of
    truth); imported lazily to avoid an import cycle
    (line_tokenization -> line_dictionary -> plasma_physics -> atomic_data)."""
    from data.line_tokenization import _ATOMIC_NUMBERS
    return int(_ATOMIC_NUMBERS[symbol])


def atomic_mass(symbol: str) -> float:
    """Standard atomic weight in u. Raises KeyError for unknown symbols."""
    try:
        return ATOMIC_MASS_AMU[symbol]
    except KeyError as exc:
        raise KeyError(f"No atomic mass for element symbol {symbol!r}") from exc


def atomic_masses(symbols: Sequence[str]) -> np.ndarray:
    """Vector of atomic weights (u) in the order of ``symbols``."""
    return np.asarray([atomic_mass(s) for s in symbols], dtype=np.float64)


def mass_to_number_fractions(
    w: np.ndarray, symbols: Sequence[str], renormalize: bool = True,
) -> np.ndarray:
    """Convert mass fractions ``w[..., E]`` to number (mole) fractions.

    x_e = (w_e / M_e) / sum_j (w_j / M_j).  Works on any leading batch shape.
    Rows that sum to zero are returned unchanged (all zeros).
    """
    w = np.asarray(w, dtype=np.float64)
    m = atomic_masses(symbols)
    x = w / m
    if renormalize:
        s = x.sum(axis=-1, keepdims=True)
        s[s == 0] = 1.0
        x = x / s
    return x


def number_to_mass_fractions(
    x: np.ndarray, symbols: Sequence[str], renormalize: bool = True,
) -> np.ndarray:
    """Convert number fractions ``x[..., E]`` to mass fractions.

    w_e = x_e M_e / sum_j x_j M_j.  Works on any leading batch shape.
    """
    x = np.asarray(x, dtype=np.float64)
    m = atomic_masses(symbols)
    w = x * m
    if renormalize:
        s = w.sum(axis=-1, keepdims=True)
        s[s == 0] = 1.0
        w = w / s
    return w
