"""
Spectral deconvolution of blended lines at a known plasma state.

A line rejected by the isolation filter is not unmeasurable — it is unresolved.
Everything needed to separate it is known: the central wavelengths of every
line in the window (the DB), the plasma temperature and electron density (the
Saha–Boltzmann solve), and the profile shape (Doppler from T and the atomic
mass, Stark as a Lorentzian, convolved with the instrument function).

Two models, both linear in their unknowns, so a window is solved by
non-negative least squares rather than by a multi-start optimiser:

``per_element`` (default)
    One unknown per element: the *relative* intensities of that element's lines
    are fixed by the Boltzmann factor at the known T (``LineSet.eps_per_n``,
    which already carries the ionisation-stage fraction).  A window with ten
    blended lines from three elements therefore has three free parameters, not
    ten, and the fit is well conditioned even on a 30-point window.
    The deconvolved area of one line is its element's coefficient times its own
    Boltzmann weight.

``per_line``
    One unknown per line, ridge-regularised.  Makes no assumption about LTE
    between lines of the same element, at the price of conditioning; useful as
    a check on the ``per_element`` result, not as the default.

``hybrid``
    The target lines (the ones whose areas are wanted) get their own unknown;
    every other line of the element in the window is tied into one
    theory-weighted column as in ``per_element``.  This is the mode to use when
    the areas feed a Boltzmann plot: under ``per_element`` the target's area is
    the window's element scale times its own Boltzmann weight at the *assumed*
    T, so its Boltzmann point is partly an echo of that T, whereas here the
    target's intensity is measured and only the background forest is modelled.

Both add a linear baseline (two free, unbounded columns) for the continuum.

Limits worth stating: the model is optically thin inside the window (no
self-absorption between the blended components), the Stark width is a single
value for every line rather than per transition, and any line missing from the
database cannot be separated out — it is absorbed by whichever component sits
nearest, which is the main failure mode to watch.

Usage:
    from cf.deconv import DeconvConfig, deconvolve_spectrum
    out = deconvolve_spectrum(wl, spectrum, targets, T, Ne, db_path, cfg)
    out.area        # [n_targets] deconvolved integrated area per target line
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.atomic_data import atomic_mass  # noqa: E402
from data.plasma_physics import (  # noqa: E402
    LineSet,
    doppler_sigma_nm,
    line_set_for_element,
    voigt_profile,
)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class DeconvConfig:
    """Window and profile settings for the deconvolution."""

    window_nm: float = 0.30          # half-width of the fitted window
    margin_nm: float = 0.60          # lines this far outside still contribute
    instrument_fwhm_nm: float = 0.047
    gamma_nm: float = 0.010          # Stark HWHM, one value for every line
    mode: str = "per_element"        # per_element | per_line | hybrid
    ridge: float = 1e-6              # per_line / hybrid only
    baseline: str = "linear"         # linear | constant | none
    min_weight: float = 1e-12        # drop components with no intensity at T


@dataclass
class DeconvResult:
    """Deconvolved areas and the diagnostics needed to judge them."""

    area: np.ndarray                 # [n_targets] integrated area per target line
    coeff: dict[str, float] = field(default_factory=dict)   # element -> scale
    n_components: int = 0            # free non-baseline columns in the window
    n_lines_in_window: int = 0       # DB lines that contributed
    rss: float = float("nan")        # residual sum of squares
    tss: float = float("nan")        # total sum of squares (about the mean)
    blend_fraction: np.ndarray | None = None   # [n_targets] 1 - own share at the centre
    # filled only when deconvolve_window(..., return_components=True); for plots
    wl: np.ndarray | None = None               # [n_pts] window axis
    y: np.ndarray | None = None                # [n_pts] measured intensities
    fit: np.ndarray | None = None              # [n_pts] total model
    baseline: np.ndarray | None = None         # [n_pts] fitted continuum
    components: dict[str, np.ndarray] = field(default_factory=dict)   # element -> curve

    @property
    def r2(self) -> float:
        return float(1.0 - self.rss / self.tss) if self.tss > 0 else float("nan")


class _LineSetCache:
    """``line_set_for_element`` memoised on (element, T, Ne) — the DB read and
    the partition functions dominate the cost otherwise."""

    def __init__(self, db_path: str, decimals: tuple[int, int] = (0, 2)):
        self.db_path = db_path
        self.decimals = decimals
        self._cache: dict[tuple[str, float, float], LineSet] = {}

    def get(self, element: str, T: float, Ne: float) -> LineSet:
        key = (element, round(float(T), self.decimals[0]),
               round(float(np.log10(max(Ne, 1.0))), self.decimals[1]))
        ls = self._cache.get(key)
        if ls is None:
            ls = line_set_for_element(element, key[1], 10.0 ** key[2], self.db_path)
            self._cache[key] = ls
        return ls


def _sigma_total(wl_nm: np.ndarray, T: float, mass_amu: float,
                 instrument_fwhm_nm: float) -> np.ndarray:
    """Gaussian width: Doppler at T convolved with the instrument function."""
    sig_dopp = doppler_sigma_nm(wl_nm, T, mass_amu)
    sig_inst = float(instrument_fwhm_nm) * FWHM_TO_SIGMA
    return np.sqrt(sig_dopp ** 2 + sig_inst ** 2)


def _baseline_columns(wl: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros((wl.size, 0), dtype=np.float64)
    ones = np.ones_like(wl)
    if kind == "constant":
        return ones[:, None]
    span = wl.max() - wl.min()
    t = (wl - wl.mean()) / (span if span > 0 else 1.0)
    return np.stack([ones, t], axis=1)


def _solve_nnls(A: np.ndarray, y: np.ndarray, n_free_tail: int,
                ridge: float | np.ndarray = 0.0) -> np.ndarray:
    """Least squares with the first columns bounded non-negative and the last
    ``n_free_tail`` (baseline) columns unbounded.  ``ridge`` is a scalar or a
    per-column vector (0 = unpenalised); it must only touch columns whose
    coefficient is an area (unit-area profiles), never the theory-weighted
    element columns whose coefficients are ~1e20 larger."""
    n = A.shape[1]
    lo = np.full(n, 0.0)
    hi = np.full(n, np.inf)
    if n_free_tail:
        lo[-n_free_tail:] = -np.inf
    r = np.broadcast_to(np.asarray(ridge, dtype=np.float64), (n,))
    if np.any(r > 0):
        A = np.vstack([A, np.diag(np.sqrt(r))])
        y = np.concatenate([y, np.zeros(n)])
    res = lsq_linear(A, y, bounds=(lo, hi), method="trf", tol=1e-10, max_iter=200)
    return res.x


def deconvolve_window(
    wl: np.ndarray,
    spectrum: np.ndarray,
    centre_nm: float,
    elements: Sequence[str],
    T: float,
    Ne: float,
    cache: _LineSetCache,
    cfg: DeconvConfig,
    targets: Sequence[tuple[str, float]] = (),
    return_components: bool = False,
) -> DeconvResult:
    """Separate the lines of one window and return the target lines' areas.

    Args:
        wl, spectrum: the measured axis and intensities (any monotonic slice)
        centre_nm: centre of the window to fit
        elements: elements allowed to contribute (the matrix + the targets)
        T, Ne: plasma state, from the Saha–Boltzmann solve
        cache: :class:`_LineSetCache`
        cfg: :class:`DeconvConfig`
        targets: (element, wavelength) pairs whose areas are wanted
    """
    m = np.abs(wl - centre_nm) <= cfg.window_nm
    if m.sum() < 5:
        return DeconvResult(area=np.full(len(targets), np.nan))
    x, y = wl[m].astype(np.float64), spectrum[m].astype(np.float64)

    cols: list[np.ndarray] = []
    col_elements: list[str] = []
    col_free: list[bool] = []          # True = one line, unit-area column (ridge applies)
    # per line: (element, wavelength, boltzmann weight, column index)
    book: list[tuple[str, float, float, int]] = []
    n_lines = 0

    for el in elements:
        ls = cache.get(el, T, Ne)
        if len(ls) == 0:
            continue
        sel = np.abs(ls.wl_nm - centre_nm) <= (cfg.window_nm + cfg.margin_nm)
        if not np.any(sel):
            continue
        wl_l = ls.wl_nm[sel]
        wgt = ls.eps_per_n[sel]
        if not np.any(wgt > cfg.min_weight):
            continue
        sigma = _sigma_total(wl_l, T, atomic_mass(el), cfg.instrument_fwhm_nm)
        prof = np.stack([voigt_profile(x - c, s, cfg.gamma_nm)
                         for c, s in zip(wl_l, sigma)], axis=1)      # [n_pts, n_lines]
        n_lines += int(sel.sum())
        if cfg.mode == "hybrid":
            # target lines get their own free scale; the rest of the element's
            # lines in the window are one theory-weighted column
            own = np.array([any(el == el_t and abs(c - wl_t) <= 0.05 for el_t, wl_t in targets)
                            for c in wl_l], dtype=bool)
        elif cfg.mode == "per_line":
            own = np.ones(wl_l.size, dtype=bool)
        else:
            own = np.zeros(wl_l.size, dtype=bool)
        if np.any(~own):
            col = prof[:, ~own] @ wgt[~own]
            if np.any(col > 0):
                idx = len(cols)
                cols.append(col)
                col_elements.append(el)
                col_free.append(False)
                for c, w in zip(wl_l[~own], wgt[~own]):
                    book.append((el, float(c), float(w), idx))
        for j in np.where(own)[0]:
            idx = len(cols)
            cols.append(prof[:, j])
            col_elements.append(el)
            col_free.append(True)
            book.append((el, float(wl_l[j]), 1.0, idx))

    if not cols:
        return DeconvResult(area=np.full(len(targets), np.nan), n_lines_in_window=n_lines)

    base = _baseline_columns(x, cfg.baseline)
    A = np.concatenate([np.stack(cols, axis=1), base], axis=1)
    ridge = np.concatenate([np.where(col_free, cfg.ridge, 0.0), np.zeros(base.shape[1])])
    coef = _solve_nnls(A, y, n_free_tail=base.shape[1], ridge=ridge)
    fit = A @ coef
    rss = float(np.sum((y - fit) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))

    # areas of the requested lines, plus how much of the peak is not theirs
    areas = np.full(len(targets), np.nan, dtype=np.float64)
    blend = np.full(len(targets), np.nan, dtype=np.float64)
    for k, (el, wl_t) in enumerate(targets):
        hit = min((b for b in book if b[0] == el),
                  key=lambda b: abs(b[1] - wl_t), default=None)
        if hit is None or abs(hit[1] - wl_t) > 0.05:
            continue
        el_h, c_h, w_h, idx = hit
        areas[k] = float(coef[idx] * w_h)
        sigma_t = float(_sigma_total(np.array([c_h]), T, atomic_mass(el_h),
                                     cfg.instrument_fwhm_nm)[0])
        own = areas[k] * float(voigt_profile(np.zeros(1), sigma_t, cfg.gamma_nm)[0])
        total = 0.0
        for el_b, c_b, w_b, idx_b in book:
            s_b = float(_sigma_total(np.array([c_b]), T, atomic_mass(el_b),
                                     cfg.instrument_fwhm_nm)[0])
            total += float(coef[idx_b] * w_b
                           * voigt_profile(np.array([c_h - c_b]), s_b, cfg.gamma_nm)[0])
        blend[k] = float(1.0 - own / total) if total > 0 else np.nan

    out = DeconvResult(
        area=areas,
        coeff={el: float(c) for el, c in zip(col_elements, coef[:len(cols)])},
        n_components=len(cols),
        n_lines_in_window=n_lines,
        rss=rss, tss=tss, blend_fraction=blend,
    )
    if return_components:
        out.wl, out.y, out.fit = x, y, fit
        out.baseline = base @ coef[len(cols):] if base.shape[1] else np.zeros_like(x)
        comp: dict[str, np.ndarray] = {}
        for j, el in enumerate(col_elements):
            curve = cols[j] * coef[j]
            comp[el] = comp[el] + curve if el in comp else curve
        out.components = comp
    return out


def deconvolve_spectrum(
    wl: np.ndarray,
    spectrum: np.ndarray,
    targets: Sequence[tuple[str, float]],
    T: float,
    Ne: float,
    db_path: str,
    elements: Sequence[str],
    cfg: DeconvConfig | None = None,
    cache: _LineSetCache | None = None,
) -> DeconvResult:
    """Deconvolve every target line of one spectrum.

    Target lines closer than ``window_nm`` to each other share a window, so a
    blend of two wanted lines is solved once, consistently.
    """
    cfg = cfg or DeconvConfig()
    cache = cache or _LineSetCache(db_path)
    order = np.argsort([t[1] for t in targets])
    areas = np.full(len(targets), np.nan, dtype=np.float64)
    blend = np.full(len(targets), np.nan, dtype=np.float64)
    n_comp = n_lines = 0
    rss = tss = 0.0
    coeff: dict[str, float] = {}

    group: list[int] = []
    for pos in list(order) + [None]:
        if pos is not None:
            if not group or abs(targets[pos][1] - targets[group[0]][1]) <= cfg.window_nm:
                group.append(int(pos))
                continue
        if group:
            centre = float(np.mean([targets[i][1] for i in group]))
            res = deconvolve_window(wl, spectrum, centre, elements, T, Ne, cache, cfg,
                                    targets=[targets[i] for i in group])
            for k, i in enumerate(group):
                areas[i] = res.area[k]
                if res.blend_fraction is not None:
                    blend[i] = res.blend_fraction[k]
            n_comp = max(n_comp, res.n_components)
            n_lines += res.n_lines_in_window
            rss += 0.0 if np.isnan(res.rss) else res.rss
            tss += 0.0 if np.isnan(res.tss) else res.tss
            coeff.update(res.coeff)
        group = [int(pos)] if pos is not None else []

    return DeconvResult(area=areas, coeff=coeff, n_components=n_comp,
                        n_lines_in_window=n_lines, rss=rss, tss=tss,
                        blend_fraction=blend)
