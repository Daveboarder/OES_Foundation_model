"""
Reference numpy implementation of the calibration-free Saha–Boltzmann solver
for ONE spectrum.  ``cf.layer.SahaBoltzmannLayer`` is a batched torch port
of exactly this function; ``scripts/check_cf_layer.py`` asserts they agree.

Inputs are raw line tokens (``data/line_tokenization.py`` layout):

    0 wavelength [nm], 1 E_i [eV], 2 E_k [eV], 3 log10 g_i, 4 log10 g_k,
    5 log10 A_k, 7 Z, 8 ion_binary (0 = I, 1 = II), 9 fitted Voigt AREA.

Line i of element e (via ``tables.z_to_elem``) in stage z contributes

    y_i = ln(area_i lambda_i / (g_k A_k)) - z ln F(T)
        = q_e - (E_k,i + z E_ion,e) beta - z eta

with beta = 1/(kT) [1/eV], eta = ln N_e and one intercept q_e per element
(q_e = ln(const * n_e l / ((1 + S10_e) U_I,e))).  The unknowns
theta = [q_1..q_E, beta, eta] come from weighted least squares with prior
rows on beta and eta and a ridge; F(T) is refreshed in a fixed-point loop.

Self-absorption (``sa_correction``): after each solve the closure gives
number fractions x, from which every line's centre optical depth is

    tau0_i = kt_i(T) * x_e r_z(T, N_e) * (N l) * phi_peak_i / NM_TO_CM,

and the next solve uses ``area_i / f(tau0_i)`` with the curve-of-growth
factor f taken from the table in ``CFTables`` (shape depends only on the
damping ratio gamma_nm / sigma_Doppler).

Closure: x_e ∝ U_I,e(T) exp(q_e) (1 + S10_e).  Elements without any used
line take their share from the seed ``C0`` (mass fractions → number
fractions): the CF elements are rescaled so that their total equals the
total ``C0`` assigns to them, seed elements keep their ``C0`` value, and the
vector is normalised.  Without ``C0`` those elements get x = 0.

Seed pre-correction (``sa_seed_init``, default on): when ``C0`` is given
and ``sa_correction`` is on, the optical depths of the *first* solve are
already evaluated at (T0, Ne0, x from C0).  The plain fixed point converges
slowly on heavily saturated line sets (contraction ≈ 0.5 per iteration,
≈ 5–10 % residual error on the majors after 3 iterations for tau0 up to
100); starting from the seed composition brings the same problem to < 1 %
in 3 iterations even with a seed perturbed by ±30 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.plasma_physics import (
    KB_EV,
    NM_TO_CM,
    doppler_sigma_nm,
    line_absorption_int,
    saha_ratio,
    saha_thermal_factor,
    voigt_peak,
)
from cf.tables import CFTables

# Token channels
CH_WL, CH_EI, CH_EK, CH_LOG_GI, CH_LOG_GK, CH_LOG_AK, CH_LOG_ITH, CH_Z, CH_ION, CH_AREA = range(10)

LN10 = float(np.log(10.0))
LOG10_NE_MIN = 10.0     # clamp for eta = ln N_e inside the loop
LOG10_NE_MAX = 22.0
TAU_FLOOR = 1e-30       # avoid 0/0 in f(tau0) for lines with x = 0

DEFAULT_CFG: dict = {
    "n_iter": 3,
    "ridge": 1e-6,
    "prior_T": 0.1,
    "prior_Ne": 0.1,
    "sa_correction": True,
    "gamma_nm": 0.01,
    "eps": 1e-7,
    "min_area": 0.0,
    "sa_seed_init": True,
    # robustness guards (a spectroscopist's rules, no learned parameters)
    "min_lines": 2,          # elements with fewer weighted lines are not solved by CF (seed / none)
    "reject_sigma": 3.0,     # drop lines whose residual exceeds k * robust sigma (0 = off)
    "reject_floor": 0.15,    # floor of the robust sigma [ln units]
}


@dataclass
class CFResult:
    """Output of :func:`saha_boltzmann_solve_np` (all arrays float64/bool)."""

    mass_fractions: np.ndarray      # [E], sums to 1 (or 0 if nothing known)
    number_fractions: np.ndarray    # [E]
    T: float                        # K
    log10_Ne: float                 # log10 cm^-3
    intercepts: np.ndarray          # [E] q_e; nan where the element has no used line
    tau0: np.ndarray                # [L] centre optical depth at the final state (0 for non-target lines)
    resid: np.ndarray               # [L] y - model at the final state (0 for invalid lines)
    used_mask: np.ndarray           # [L] bool, weight > 0 and valid
    n_lines_used: np.ndarray        # [E] int64
    censored: np.ndarray            # [E] bool
    source: np.ndarray              # [E] str: 'cf' | 'seed' | 'none'
    n_iter: int = 0
    T_history: np.ndarray | None = None       # [n_iter]
    log10_Ne_history: np.ndarray | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (kept separate so the torch port mirrors them one by one)
# ─────────────────────────────────────────────────────────────────────────────
def curve_of_growth_table(tau0: np.ndarray, a: np.ndarray, tables: CFTables) -> np.ndarray:
    """f(tau0) from the ``CFTables`` shape table; ``a`` = damping ratio
    gamma/sigma per line.  Vectorised over the leading shape of ``tau0``."""
    tau0 = np.asarray(tau0, dtype=np.float64)
    log_a = np.log(np.asarray(a, dtype=np.float64))
    la = tables.cog_log_a
    d = la[1] - la[0]
    pos = (np.clip(log_a, la[0], la[-1]) - la[0]) / d
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, la.size - 2)
    frac = pos - i0
    phi_hat = tables.cog_phi_hat[i0] * (1.0 - frac)[..., None] + tables.cog_phi_hat[i0 + 1] * frac[..., None]
    tau_safe = np.maximum(tau0, TAU_FLOOR)[..., None]
    num = (-np.expm1(-tau_safe * phi_hat)) @ tables.cog_w
    den = phi_hat @ tables.cog_w
    return np.clip(num / (tau_safe[..., 0] * den), 0.0, 1.0)


def _lower_median(v: np.ndarray) -> float:
    """Lower median (element at index (n-1)//2 of the sorted values), matching
    ``torch.nanmedian`` so the numpy reference and the torch layer reject the
    same lines."""
    v = np.sort(np.asarray(v, dtype=np.float64).reshape(-1))
    return float(v[(v.size - 1) // 2]) if v.size else float("nan")


def _solve_weighted_ls(A: np.ndarray, w: np.ndarray, y: np.ndarray, E: int,
                       beta0: float, eta0: float, prior_T: float, prior_Ne: float,
                       ridge: float) -> np.ndarray:
    """theta = argmin sum_i w_i (y_i - A_i theta)^2 + prior_T (beta - beta0)^2
    + prior_Ne (eta - eta0)^2 + ridge |theta|^2."""
    M = (A * w[:, None]).T @ A
    b = A.T @ (w * y)
    diag = np.full(E + 2, ridge, dtype=np.float64)
    diag[E] += prior_T
    diag[E + 1] += prior_Ne
    M[np.diag_indices_from(M)] += diag
    b[E] += prior_T * beta0
    b[E + 1] += prior_Ne * eta0
    return np.linalg.solve(M, b)


def saha_boltzmann_solve_np(
    tokens: np.ndarray,
    fit_valid: np.ndarray,
    weights: np.ndarray,
    tables: CFTables,
    C0: np.ndarray | None = None,
    T0: float = 10000.0,
    log10_Ne0: float = 17.0,
    log10_Nl0: float = 16.0,
    n_iter: int = 3,
    ridge: float = 1e-6,
    prior_T: float = 0.1,
    prior_Ne: float = 0.1,
    sa_correction: bool = True,
    gamma_nm: float = 0.01,
    eps: float = 1e-7,
    isolation: np.ndarray | None = None,
    min_area: float = 0.0,
    sa_seed_init: bool = True,
    min_lines: int = 2,
    reject_sigma: float = 3.0,
    reject_floor: float = 0.15,
) -> CFResult:
    """Calibration-free Saha–Boltzmann solve for one spectrum.

    Args:
        tokens: ``[L, 14]`` raw line tokens
        fit_valid: ``[L]`` 0/1 Voigt-fit validity
        weights: ``[L]`` per-line reliability weights ≥ 0 (multiplied by
            ``fit_valid`` and by the validity of the token internally)
        tables: :class:`cf.tables.CFTables`
        C0: ``[E]`` seed mass fractions for elements without used lines
            (``None`` → those elements get x = 0, source ``'none'``)
        T0, log10_Ne0: initial guesses; also the centres of the priors
        log10_Nl0: log10 of the heavy-particle column density N·l [cm^-2]
            used for the optical depths
        n_iter: fixed-point iterations (F(T) refresh + self-absorption)
        ridge, prior_T, prior_Ne: regularisation of the least squares
        sa_correction: apply the curve-of-growth correction
        gamma_nm: Lorentz HWHM [nm] assumed for every line profile
        eps: numerical floor for normalisations and f(tau0)
        isolation: optional ``[L]`` factor (e.g. isolation score) multiplied
            into the weights
        min_area: lines with area ≤ ``min_area`` are never used
        sa_seed_init: evaluate the optical depths of the first solve at
            (T0, Ne0, x from ``C0``) when ``C0`` is given (see module doc)
        min_lines: elements with fewer than ``min_lines`` weighted lines are
            not solved by CF (their lines are dropped; value from the seed)
        reject_sigma: after every solve but the last, lines whose residual
            deviates from the median by more than ``reject_sigma`` robust
            sigmas (1.4826·MAD, floored at ``reject_floor``) are dropped and
            the gating is re-applied (0 disables)
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] < 10:
        raise ValueError(f"tokens must be [L, >=10], got {tokens.shape}")
    L = tokens.shape[0]
    E = tables.n_elements

    wl = tokens[:, CH_WL]
    Ei = tokens[:, CH_EI]
    Ek = tokens[:, CH_EK]
    gk = np.power(10.0, tokens[:, CH_LOG_GK])
    Ak = np.power(10.0, tokens[:, CH_LOG_AK])
    ln_gkA = (tokens[:, CH_LOG_GK] + tokens[:, CH_LOG_AK]) * LN10
    z = np.clip(np.rint(tokens[:, CH_ION]), 0.0, 1.0)
    area = tokens[:, CH_AREA]

    e_idx = tables.elem_index(tokens[:, CH_Z])
    is_target = e_idx >= 0
    e_safe = np.where(is_target, e_idx, 0)
    valid = (
        (np.asarray(fit_valid, dtype=np.float64) > 0.5)
        & is_target
        & np.isfinite(area) & (area > 0.0) & (area > float(min_area))
        & (wl > 0.0)
    )
    w = np.asarray(weights, dtype=np.float64) * valid
    if isolation is not None:
        w = w * np.asarray(isolation, dtype=np.float64)
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)

    def _gate(w: np.ndarray):
        """Zero the lines of elements with fewer than ``min_lines`` weighted lines."""
        used = w > 0.0
        n = np.bincount(e_safe[used], minlength=E)
        ok = n >= max(int(min_lines), 1)
        w = w * ok[e_safe]
        used = w > 0.0
        n = np.bincount(e_safe[used], minlength=E).astype(np.int64)
        return w, used, n, n > 0

    w, used, n_lines_used, has_lines = _gate(w)

    # per-line element constants
    E_ion_line = tables.E_ion[e_safe]
    mass_line = tables.mass_amu[e_safe]
    Eeff = Ek + z * E_ion_line
    onehot = np.zeros((L, E), dtype=np.float64)
    onehot[np.arange(L), e_safe] = is_target
    A = np.concatenate([onehot, -Eeff[:, None], -z[:, None]], axis=1)   # [L, E+2]

    area_safe = np.where(valid, area, 1.0)
    y_base = np.log(area_safe * np.where(wl > 0, wl, 1.0)) - ln_gkA

    # seed composition as number fractions
    if C0 is not None:
        C0 = np.asarray(C0, dtype=np.float64).reshape(E)
        x0 = C0 / tables.mass_amu
        x0 = x0 / max(float(x0.sum()), eps)
        C0_mass = C0
    else:
        x0 = np.zeros(E, dtype=np.float64)
        C0_mass = np.zeros(E, dtype=np.float64)

    T0 = float(tables.clamp_T(T0))
    beta0 = 1.0 / (KB_EV * T0)
    eta0 = float(np.clip(log10_Ne0, LOG10_NE_MIN, LOG10_NE_MAX)) * LN10
    Nl = float(10.0 ** float(log10_Nl0))

    T = T0
    lnNe = eta0
    ln_f = np.zeros(L, dtype=np.float64)
    q = np.zeros(E, dtype=np.float64)
    x = np.zeros(E, dtype=np.float64)
    tau0 = np.zeros(L, dtype=np.float64)
    resid = np.zeros(L, dtype=np.float64)
    T_hist, ne_hist = [], []
    z_int = z.astype(np.int64)

    state = {"has_lines": has_lines}

    def closure(T: float, lnNe: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Number fractions x [E] and Saha ratios S10 [E] at (T, Ne, q)."""
        has_lines = state["has_lines"]
        U_T = tables.partition_functions(T)                               # [E, 2]
        S10 = saha_ratio(T, np.exp(lnNe), U_T[:, 0], U_T[:, 1], tables.E_ion)
        x_tilde = np.where(has_lines, U_T[:, 0] * np.exp(q) * (1.0 + S10), 0.0)
        if C0 is not None:
            scale = max(float((x0 * has_lines).sum()), eps) / max(float(x_tilde.sum()), eps)
            x = np.where(has_lines, x_tilde * scale, x0)
        else:
            x = x_tilde
        return x / max(float(x.sum()), eps), S10

    def optical_depth(T: float, x: np.ndarray, S10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centre optical depth tau0 [L] and damping ratio a [L] of every line."""
        U_T = tables.partition_functions(T)
        sigma = doppler_sigma_nm(wl, T, mass_line)                        # [L]
        a = gamma_nm / sigma
        phi_peak = voigt_peak(sigma, gamma_nm)                            # per nm
        kt = line_absorption_int(wl, Ak, gk, Ei, Ek, T, U_T[e_safe, z_int])
        r_line = np.where(z > 0.5, (S10 / (1.0 + S10))[e_safe], (1.0 / (1.0 + S10))[e_safe])
        tau0 = np.where(is_target, kt * x[e_safe] * r_line * Nl * phi_peak / NM_TO_CM, 0.0)
        return tau0, a

    def sa_log_factor(tau0: np.ndarray, a: np.ndarray) -> np.ndarray:
        f = np.maximum(curve_of_growth_table(tau0, a, tables), eps)
        return np.where(valid, np.log(f), 0.0)

    if sa_correction and sa_seed_init and C0 is not None:
        # optical depths of the first solve from the seed composition
        U_T = tables.partition_functions(T)
        S10 = saha_ratio(T, np.exp(lnNe), U_T[:, 0], U_T[:, 1], tables.E_ion)
        tau0, a = optical_depth(T, x0, S10)
        ln_f = sa_log_factor(tau0, a)

    n_iter = int(n_iter)
    for it in range(n_iter):
        # ── weighted least squares ──────────────────────────────────────
        y = y_base - ln_f - z * np.log(saha_thermal_factor(T))
        theta = _solve_weighted_ls(A, w, y, E, beta0, eta0, prior_T, prior_Ne, ridge)
        q = theta[:E]
        beta = theta[E]
        eta = theta[E + 1]
        T = float(tables.clamp_T(1.0 / (KB_EV * beta) if beta > 0 else tables.T_grid[-1]))
        lnNe = float(np.clip(eta, LOG10_NE_MIN * LN10, LOG10_NE_MAX * LN10))
        resid = np.where(valid, y - A @ theta, 0.0)
        T_hist.append(T)
        ne_hist.append(lnNe / LN10)

        # ── robust line rejection (not after the last solve) ───────────
        if reject_sigma > 0 and it < n_iter - 1 and int(used.sum()) >= 4:
            r = resid[used]
            med = _lower_median(r)                       # same convention as torch.nanmedian
            sig = max(1.4826 * _lower_median(np.abs(r - med)), float(reject_floor))
            bad = used & (np.abs(resid - med) > float(reject_sigma) * sig)
            if bad.any():
                w, used, n_lines_used, has_lines = _gate(w * (~bad))
                state["has_lines"] = has_lines

        # ── closure, optical depth, curve of growth ─────────────────────
        x, S10 = closure(T, lnNe, q)
        tau0, a = optical_depth(T, x, S10)
        if sa_correction:
            ln_f = sa_log_factor(tau0, a)

    # ── outputs ────────────────────────────────────────────────────────
    mass = x * tables.mass_amu
    mass = mass / max(float(mass.sum()), eps)
    intercepts = np.where(has_lines, q, np.nan)
    censored = (mass < tables.lod) | (~has_lines & (C0_mass < tables.lod))
    source = np.where(has_lines, "cf", "seed" if C0 is not None else "none").astype("<U4")

    return CFResult(
        mass_fractions=mass,
        number_fractions=x,
        T=float(T),
        log10_Ne=float(lnNe / LN10),
        intercepts=intercepts,
        tau0=tau0,
        resid=resid,
        used_mask=used,
        n_lines_used=n_lines_used,
        censored=censored,
        source=source,
        n_iter=int(n_iter),
        T_history=np.asarray(T_hist, dtype=np.float64),
        log10_Ne_history=np.asarray(ne_hist, dtype=np.float64),
    )
