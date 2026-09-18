"""
Precomputed lookup tables for the calibration-free Saha–Boltzmann solver.

``CFTables`` bundles everything the solver needs that does *not* depend on the
spectrum: per-element ionisation energies, atomic masses, detection limits,
the neutral/ion partition functions tabulated on a uniform temperature grid,
and the curve-of-growth profile table used for the self-absorption
correction.  ``build_cf_tables`` fills it from the SQLite line database
(``PartF_var`` / ``E_ion`` tables via ``data.libs_pipeline``) and
``config/element_lod.yaml``; ``CFTables.to_torch`` mirrors it as float64
tensors for ``cf.layer.SahaBoltzmannLayer``.

Partition functions
-------------------
``U[e, s, t]`` = sum_levels g exp(-E / (k T_grid[t])) for element ``e`` and
stage ``s`` (0 = I, 1 = II), identical to
``data.libs_pipeline.partition_function_cached`` evaluated at ``T_grid[t]``
(same ``kb_eV`` constant), but vectorised over the grid so a whole element
costs one matrix product.  Both solvers evaluate U(T) by linear interpolation
on this grid with T clamped to its range.

Curve of growth
---------------
The self-absorption factor of a homogeneous slab,

    f(tau0) = int (1 - exp(-tau0 phi_hat(u))) du / (tau0 int phi_hat(u) du),

depends on the Voigt *shape* only through the damping ratio a = gamma/sigma
once the abscissa is measured in units of the FWHM (phi_hat = phi/phi_peak).
``cog_phi_hat[k, j]`` therefore stores phi_hat on a fixed abscissa grid
``cog_u[j]`` (in FWHM units) for a log-spaced damping-ratio grid
``exp(cog_log_a[k])``; ``cog_w[j]`` are the trapezoid weights so the integral
is a dot product.  The abscissa is a sinh-mapped grid (dense in the core,
reaching ``cog_half_width`` FWHM in the wings) — a *uniform* 129-point grid
over ±20 FWHM under-estimates f by 11–22 % for tau0 ≥ 30 on Lorentz-dominated
profiles because the wings are truncated, whereas the sinh grid with the same
number of points agrees with a 2·10^6-point reference to < 3·10^-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from data.atomic_data import atomic_masses, atomic_number
from data.libs_pipeline import _get_eion, _load_partf
from data.plasma_physics import KB_EV, voigt_fwhm, voigt_peak, voigt_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOD_CONFIG = _REPO_ROOT / "config" / "element_lod.yaml"

T_GRID_MIN = 3000.0     # K
T_GRID_MAX = 30000.0    # K
T_GRID_STEP = 100.0     # K
N_Z = 120               # z_to_elem covers Z = 0 .. 119

# Curve-of-growth table defaults
COG_N_GRID = 129            # abscissa points (FWHM units, sinh-mapped)
COG_HALF_WIDTH = 2000.0     # |u| max in FWHM
COG_CORE_SCALE = 0.5        # sinh scale: spacing ~ core_scale * ds near u = 0
COG_LOG_A_MIN = np.log(1e-4)
COG_LOG_A_MAX = np.log(1e4)
COG_N_A = 1201


@dataclass
class CFTables:
    """Spectrum-independent tables for the CF solver (numpy, float64).

    Attributes:
        element_names: target elements, order of every ``[E]`` axis
        z_to_elem: int64 ``[120]``; ``z_to_elem[Z]`` = element index, -1 if
            ``Z`` is not a target element
        E_ion: first ionisation energy [eV] ``[E]``
        mass_amu: standard atomic weight [u] ``[E]``
        T_grid: ``[nT]`` uniform, 3000..30000 K step 100 K
        U: partition functions ``[E, 2, nT]`` (index 1: 0 = stage I, 1 = stage II)
        lod: limit of detection as mass fraction ``[E]``
        cog_u: curve-of-growth abscissa in FWHM units ``[nG]``
        cog_w: trapezoid weights for ``cog_u`` ``[nG]``
        cog_log_a: ln(damping ratio) grid ``[nA]`` (uniform)
        cog_phi_hat: peak-normalised Voigt shape ``[nA, nG]``
    """

    element_names: list[str]
    z_to_elem: np.ndarray
    E_ion: np.ndarray
    mass_amu: np.ndarray
    T_grid: np.ndarray
    U: np.ndarray
    lod: np.ndarray
    cog_u: np.ndarray
    cog_w: np.ndarray
    cog_log_a: np.ndarray
    cog_phi_hat: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    # ── convenience ────────────────────────────────────────────────────────
    @property
    def n_elements(self) -> int:
        return len(self.element_names)

    @property
    def T_step(self) -> float:
        return float(self.T_grid[1] - self.T_grid[0])

    def elem_index(self, Z: np.ndarray) -> np.ndarray:
        """Element index per atomic number (``-1`` for non-target elements)."""
        Z = np.asarray(np.rint(Z), dtype=np.int64)
        out = np.full(Z.shape, -1, dtype=np.int64)
        ok = (Z >= 0) & (Z < self.z_to_elem.size)
        out[ok] = self.z_to_elem[Z[ok]]
        return out

    def clamp_T(self, T):
        return np.clip(np.asarray(T, dtype=np.float64), self.T_grid[0], self.T_grid[-1])

    def partition_functions(self, T: float) -> np.ndarray:
        """U(T) for every element and both stages, ``[E, 2]``, by linear
        interpolation on ``T_grid`` (T clamped to the grid)."""
        T = float(self.clamp_T(T))
        pos = (T - self.T_grid[0]) / self.T_step
        i0 = int(min(max(np.floor(pos), 0), self.T_grid.size - 2))
        frac = pos - i0
        return self.U[:, :, i0] * (1.0 - frac) + self.U[:, :, i0 + 1] * frac

    def to_torch(self, device=None) -> "CFTablesTorch":
        import torch

        def t(a, dtype=torch.float64):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

        return CFTablesTorch(
            element_names=list(self.element_names),
            z_to_elem=t(self.z_to_elem, torch.int64),
            E_ion=t(self.E_ion),
            mass_amu=t(self.mass_amu),
            T_grid=t(self.T_grid),
            U=t(self.U),
            lod=t(self.lod),
            cog_u=t(self.cog_u),
            cog_w=t(self.cog_w),
            cog_log_a=t(self.cog_log_a),
            cog_phi_hat=t(self.cog_phi_hat),
            meta=dict(self.meta),
        )


@dataclass
class CFTablesTorch:
    """``CFTables`` as float64 torch tensors (``z_to_elem`` int64)."""

    element_names: list[str]
    z_to_elem: Any
    E_ion: Any
    mass_amu: Any
    T_grid: Any
    U: Any
    lod: Any
    cog_u: Any
    cog_w: Any
    cog_log_a: Any
    cog_phi_hat: Any
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_elements(self) -> int:
        return len(self.element_names)


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────
def load_lod_vector(element_names: Sequence[str], lod_config_path: str | Path | None = None) -> np.ndarray:
    """Per-element LOD (mass fraction) aligned to ``element_names``.

    Same semantics as ``train_finetune.build_lod_vector``: values from
    ``limits_of_detection`` in ``config/element_lod.yaml``, ``default_lod``
    for unlisted elements.  Returns float64 ``[E]``.
    """
    path = Path(lod_config_path) if lod_config_path is not None else DEFAULT_LOD_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    default_lod = float(cfg.get("default_lod", 1.0e-4))
    table = cfg.get("limits_of_detection", {}) or {}
    return np.asarray([float(table.get(name, default_lod)) for name in element_names], dtype=np.float64)


def partition_function_grid(element: str, db_path: str, T_grid: np.ndarray) -> np.ndarray:
    """``[2, nT]`` partition functions of ``element`` on ``T_grid`` from the
    ``PartF_var`` levels (same formula/constant as
    ``data.libs_pipeline.partition_function_cached``)."""
    gi_I, Ei_I, gi_II, Ei_II = _load_partf(element, db_path)
    inv_kT = 1.0 / (KB_EV * np.asarray(T_grid, dtype=np.float64))     # [nT]
    out = np.zeros((2, T_grid.size), dtype=np.float64)
    if gi_I.size:
        out[0] = gi_I @ np.exp(-np.outer(Ei_I, inv_kT))
    if gi_II.size:
        out[1] = gi_II @ np.exp(-np.outer(Ei_II, inv_kT))
    return out


def sinh_grid(n: int, half_width: float, core_scale: float) -> np.ndarray:
    """Symmetric sinh-mapped abscissa: ``u = c sinh(s)``, ``s`` uniform on
    ``[-asinh(half_width/c), +asinh(half_width/c)]``."""
    s_max = np.arcsinh(half_width / core_scale)
    s = np.linspace(-s_max, s_max, n)
    return core_scale * np.sinh(s)


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Weights ``w`` such that ``sum(w * f) == np.trapezoid(f, x)``."""
    x = np.asarray(x, dtype=np.float64)
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def build_cog_table(
    n_grid: int = COG_N_GRID,
    half_width: float = COG_HALF_WIDTH,
    core_scale: float = COG_CORE_SCALE,
    n_a: int = COG_N_A,
    log_a_min: float = COG_LOG_A_MIN,
    log_a_max: float = COG_LOG_A_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curve-of-growth shape table ``(cog_u, cog_w, cog_log_a, cog_phi_hat)``.

    ``cog_phi_hat[k]`` is the Voigt profile with damping ratio
    ``a_k = exp(cog_log_a[k])`` (sigma = 1, gamma = a_k), evaluated at
    ``cog_u * FWHM(a_k)`` and divided by its peak — i.e. exactly the
    ``phi_hat`` of ``data.plasma_physics.curve_of_growth_factor`` on a
    non-uniform abscissa.
    """
    u = sinh_grid(n_grid, half_width, core_scale)
    w = trapezoid_weights(u)
    log_a = np.linspace(log_a_min, log_a_max, n_a)
    a = np.exp(log_a)                                        # gamma with sigma = 1
    width = voigt_fwhm(1.0, a)                               # [nA]
    dl = u[None, :] * width[:, None]                         # [nA, nG]
    phi = voigt_profile(dl, 1.0, a[:, None])
    phi_hat = phi / voigt_peak(1.0, a)[:, None]
    return u, w, log_a, phi_hat


def build_cf_tables(
    element_names: Sequence[str],
    db_path: str,
    lod_config_path: str | Path | None = None,
    T_grid: np.ndarray | None = None,
    cog_kwargs: dict[str, Any] | None = None,
) -> CFTables:
    """Assemble ``CFTables`` for ``element_names`` from the line database.

    Args:
        element_names: target elements (order defines every ``[E]`` axis)
        db_path: SQLite DB with ``E_ion`` and ``PartF_var`` tables
        lod_config_path: ``config/element_lod.yaml`` (default: repo copy)
        T_grid: override the temperature grid (uniform, ascending)
        cog_kwargs: overrides for :func:`build_cog_table`

    Raises:
        ValueError if an element has a vanishing partition function anywhere
        on the grid (the solver takes ``ln U`` and ``U_II / U_I``).
    """
    element_names = [str(e) for e in element_names]
    if len(set(element_names)) != len(element_names):
        raise ValueError("element_names must be unique")
    db_path = str(db_path)
    if T_grid is None:
        T_grid = np.arange(T_GRID_MIN, T_GRID_MAX + 0.5 * T_GRID_STEP, T_GRID_STEP, dtype=np.float64)
    T_grid = np.asarray(T_grid, dtype=np.float64)
    if T_grid.ndim != 1 or T_grid.size < 2 or not np.allclose(np.diff(T_grid), T_grid[1] - T_grid[0]):
        raise ValueError("T_grid must be 1-D, uniform and ascending")

    E = len(element_names)
    z_to_elem = np.full(N_Z, -1, dtype=np.int64)
    E_ion = np.zeros(E, dtype=np.float64)
    U = np.zeros((E, 2, T_grid.size), dtype=np.float64)
    for i, name in enumerate(element_names):
        z = atomic_number(name)
        if not 0 <= z < N_Z:
            raise ValueError(f"atomic number {z} of {name} outside z_to_elem range")
        z_to_elem[z] = i
        E_ion[i] = float(_get_eion(name, db_path))
        U[i] = partition_function_grid(name, db_path, T_grid)
    bad = [element_names[i] for i in range(E) if not np.all(U[i] > 0)]
    if bad:
        raise ValueError(f"Vanishing partition function on the T grid for: {bad}")

    cog_u, cog_w, cog_log_a, cog_phi_hat = build_cog_table(**(cog_kwargs or {}))
    return CFTables(
        element_names=element_names,
        z_to_elem=z_to_elem,
        E_ion=E_ion,
        mass_amu=atomic_masses(element_names),
        T_grid=T_grid,
        U=U,
        lod=load_lod_vector(element_names, lod_config_path),
        cog_u=cog_u,
        cog_w=cog_w,
        cog_log_a=cog_log_a,
        cog_phi_hat=cog_phi_hat,
        meta={
            "db_path": db_path,
            "lod_config_path": str(lod_config_path or DEFAULT_LOD_CONFIG),
            "cog": {"n_grid": int(cog_u.size), "n_a": int(cog_log_a.size),
                    **(cog_kwargs or {})},
        },
    )
