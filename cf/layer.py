"""
``SahaBoltzmannLayer`` — batched, differentiable torch port of
``cf.solver_np.saha_boltzmann_solve_np``.

The layer has **no trainable parameters** (tables are buffers) and runs in
float64 with autocast disabled, so it can sit inside a bf16 training graph.
It is differentiable w.r.t. the per-line ``weights`` and the initial guesses
``T0``, ``log10_Ne0``, ``log10_Nl0`` (through the priors, the F(T) term of the
first iteration, the seed pre-correction and the optical depths).  Every
step mirrors the numpy reference one-to-one — same constants, same
partition-function interpolation, same curve-of-growth table — so the two
agree to round-off (``scripts/check_cf_layer.py``).

Numerical safety: invalid lines (fit failed, non-target element, area ≤ 0)
get weight 0 and their token values are replaced by harmless constants
before any log, so no NaN can enter the graph; ``torch.where`` (not
multiplication by a mask) is used wherever a masked value could be
non-finite.
"""

from __future__ import annotations

import math
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.atomic_data import AMU_G
from data.libs_pipeline import _EV_TO_ERG, _H, _KB, _ME
from data.plasma_physics import C_CGS, KB_EV, NM_TO_CM
from cf.solver_np import (
    CH_AREA, CH_EI, CH_EK, CH_ION, CH_LOG_AK, CH_LOG_GK, CH_R2, CH_WL, CH_Z,
    DEFAULT_CFG, LN10, LOG10_NE_MAX, LOG10_NE_MIN, TAU_FLOOR,
)
from cf.tables import CFTables

# Constants written exactly as data.plasma_physics evaluates them
_LN_F_CONST = math.log(2.0) + 1.5 * math.log(2.0 * math.pi * _ME * _KB / (_H ** 2))   # ln F(T) = const + 1.5 ln T
_EV_OVER_KB = _EV_TO_ERG / _KB              # K/eV in the Boltzmann exponents of plasma_physics
_KT_PREFACTOR = 1.0 / (8.0 * math.pi * C_CGS)
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _nanmedian_lower(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Lower median along ``dim``, ignoring NaNs.

    Same value as ``torch.nanmedian(x, dim).values`` (and the numpy solver's
    ``_lower_median``), but built from a stable ``sort``: the CUDA median
    kernel has no deterministic implementation, and both trainers run the
    Lightning Trainer with ``deterministic=True``.
    """
    nan = torch.isnan(x)
    n = (~nan).sum(dim=dim, keepdim=True)
    filled = torch.where(nan, torch.full_like(x, torch.finfo(x.dtype).max), x)
    xs, _ = torch.sort(filled, dim=dim, stable=True)
    out = xs.gather(dim, torch.clamp(n - 1, min=0) // 2)
    out = torch.where(n > 0, out, torch.full_like(out, float("nan")))
    return out.squeeze(dim)


class SahaBoltzmannLayer(nn.Module):
    """Calibration-free Saha–Boltzmann solver as a parameter-free module.

    Args:
        tables: :class:`cf.tables.CFTables` (numpy); copied into buffers
        cfg: optional overrides of ``cf.solver_np.DEFAULT_CFG`` keys
            ``n_iter, ridge, prior_T, prior_Ne, sa_correction, gamma_nm, eps,
            min_area, sa_seed_init`` plus ``use_isolation`` (default False:
            multiply the weights by the ``isolation_score`` buffer)
        line_dict_path: optional ``line_dict_*.h5``; its ``isolation_score``
            and ``forced`` datasets become buffers (ones/zeros when absent)
    """

    def __init__(self, tables: CFTables, cfg: dict[str, Any] | None = None,
                 line_dict_path: str | None = None):
        super().__init__()
        cfg = {**DEFAULT_CFG, "use_isolation": False, **(cfg or {})}
        unknown = set(cfg) - set(DEFAULT_CFG) - {"use_isolation"}
        if unknown:
            raise ValueError(f"Unknown SahaBoltzmannLayer cfg keys: {sorted(unknown)}")
        self.cfg = cfg
        self.n_iter = int(cfg["n_iter"])
        self.ridge = float(cfg["ridge"])
        self.prior_T = float(cfg["prior_T"])
        self.prior_Ne = float(cfg["prior_Ne"])
        self.sa_correction = bool(cfg["sa_correction"])
        self.sa_seed_init = bool(cfg["sa_seed_init"])
        self.gamma_nm = float(cfg["gamma_nm"])
        self.eps = float(cfg["eps"])
        self.min_area = float(cfg["min_area"])
        self.use_isolation = bool(cfg["use_isolation"])
        self.min_lines = max(int(cfg["min_lines"]), 1)
        self.seed_in_closure = bool(cfg.get("seed_in_closure", False))
        self.single_line_min_isolation = float(cfg.get("single_line_min_isolation", 0.0))
        self.single_line_r2_min = float(cfg.get("single_line_r2_min", 0.0))
        self.single_line_prior = float(cfg.get("single_line_prior", 0.0))
        self.reject_sigma = float(cfg["reject_sigma"])
        self.reject_floor = float(cfg["reject_floor"])
        self.element_names = list(tables.element_names)
        self.n_elements = len(self.element_names)
        self.line_dict_path = line_dict_path

        f64 = torch.float64
        self.register_buffer("E_ion", torch.as_tensor(tables.E_ion, dtype=f64))
        self.register_buffer("mass_amu", torch.as_tensor(tables.mass_amu, dtype=f64))
        self.register_buffer("T_grid", torch.as_tensor(tables.T_grid, dtype=f64))
        self.register_buffer("U", torch.as_tensor(tables.U, dtype=f64))
        self.register_buffer("lod", torch.as_tensor(tables.lod, dtype=f64))
        self.register_buffer("z_to_elem", torch.as_tensor(tables.z_to_elem, dtype=torch.int64))
        self.register_buffer("cog_w", torch.as_tensor(tables.cog_w, dtype=f64))
        self.register_buffer("cog_log_a", torch.as_tensor(tables.cog_log_a, dtype=f64))
        self.register_buffer("cog_phi_hat", torch.as_tensor(tables.cog_phi_hat, dtype=f64))
        self.T_min = float(tables.T_grid[0])
        self.T_max = float(tables.T_grid[-1])
        self.T_step = float(tables.T_grid[1] - tables.T_grid[0])
        self.n_T = int(tables.T_grid.size)
        self.log_a0 = float(tables.cog_log_a[0])
        self.log_a1 = float(tables.cog_log_a[-1])
        self.d_log_a = float(tables.cog_log_a[1] - tables.cog_log_a[0])
        self.n_a = int(tables.cog_log_a.size)

        iso, forced = None, None
        if line_dict_path is not None:
            with h5py.File(line_dict_path, "r") as f:
                n_lines = int(f.attrs.get("n_lines", f["central_wavelength"].shape[0]))
                iso = f["isolation_score"][:].astype(np.float64) if "isolation_score" in f else np.ones(n_lines)
                forced = f["forced"][:].astype(np.float64) if "forced" in f else np.zeros(n_lines)
        if iso is not None:
            self.register_buffer("isolation_score", torch.as_tensor(iso, dtype=f64))
            self.register_buffer("forced", torch.as_tensor(forced, dtype=f64))
        else:
            self.isolation_score = None
            self.forced = None

    # ── helpers (each mirrors a numpy step) ──────────────────────────────
    def _clamp_T(self, T: torch.Tensor) -> torch.Tensor:
        return T.clamp(self.T_min, self.T_max)

    def _partition_functions(self, T: torch.Tensor) -> torch.Tensor:
        """U(T) ``[B, E, 2]`` by linear interpolation on the grid."""
        pos = (self._clamp_T(T) - self.T_min) / self.T_step                # [B]
        i0 = pos.floor().long().clamp(0, self.n_T - 2)
        frac = (pos - i0.to(pos.dtype))[:, None, None]
        U0 = self.U[:, :, i0].permute(2, 0, 1)                             # [B, E, 2]
        U1 = self.U[:, :, i0 + 1].permute(2, 0, 1)
        return U0 * (1.0 - frac) + U1 * frac

    @staticmethod
    def _ln_saha_thermal_factor(T: torch.Tensor) -> torch.Tensor:
        return _LN_F_CONST + 1.5 * torch.log(T)

    def _saha_ratio(self, T: torch.Tensor, lnNe: torch.Tensor, U_T: torch.Tensor) -> torch.Tensor:
        """S10 ``[B, E]`` = U_II/(Ne U_I) F(T) exp(-E_ion/kT) (plasma_physics constants)."""
        lnS = (torch.log(U_T[:, :, 1]) - torch.log(U_T[:, :, 0]) - lnNe[:, None]
               + self._ln_saha_thermal_factor(T)[:, None]
               - self.E_ion[None, :] * _EV_OVER_KB / T[:, None])
        return torch.exp(lnS)

    def _curve_of_growth(self, tau0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """f(tau0) ``[B, L]`` from the shape table (piecewise-linear in ln a, so
        the weak dependence of the Voigt shape on T stays differentiable)."""
        log_a = torch.log(a).clamp(self.log_a0, self.log_a1)
        pos = (log_a - self.log_a0) / self.d_log_a
        i0 = pos.floor().long().clamp(0, self.n_a - 2)
        frac = (pos - i0.to(pos.dtype))[..., None]
        phi_hat = self.cog_phi_hat[i0] * (1.0 - frac) + self.cog_phi_hat[i0 + 1] * frac   # [B, L, nG]
        tau_safe = tau0.clamp(min=TAU_FLOOR)[..., None]
        num = (-torch.expm1(-tau_safe * phi_hat)) @ self.cog_w
        den = phi_hat @ self.cog_w
        return (num / (tau_safe[..., 0] * den)).clamp(0.0, 1.0)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(
        self,
        tokens: torch.Tensor,
        fit_valid: torch.Tensor,
        weights: torch.Tensor,
        C0: torch.Tensor | None = None,
        T0: torch.Tensor | None = None,
        log10_Ne0: torch.Tensor | None = None,
        log10_Nl0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Solve a batch of spectra.

        Args:
            tokens: ``[B, L, 14]`` raw line tokens (any float dtype)
            fit_valid: ``[B, L]`` 0/1
            weights: ``[B, L]`` per-line weights in [0, 1] (masked again here)
            C0: ``[B, E]`` seed mass fractions or ``None`` (uniform 1/E)
            T0, log10_Ne0, log10_Nl0: ``[B]`` initial guesses or ``None``
                (10000 K, 17.0, 16.0)

        Returns: dict with ``concentrations`` (mass fractions ``[B, E]``),
        ``number_fractions``, ``T``, ``log10_Ne``, ``intercepts`` (nan where
        the element has no used line), ``tau0`` ``[B, L]``, ``resid``,
        ``n_lines_used``, ``censored`` (bool), ``used_mask`` (bool),
        ``has_lines`` (bool) — all float64 except the bool/int ones.
        """
        device_type = tokens.device.type if tokens.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return self._forward(tokens, fit_valid, weights, C0, T0, log10_Ne0, log10_Nl0)

    def _forward(self, tokens, fit_valid, weights, C0, T0, log10_Ne0, log10_Nl0):
        f64 = torch.float64
        dev = tokens.device
        tokens = tokens.to(f64)
        B, L, _ = tokens.shape
        E = self.n_elements
        eps = self.eps

        wl = tokens[..., CH_WL]
        Ei = tokens[..., CH_EI]
        Ek = tokens[..., CH_EK]
        gk = torch.pow(10.0, tokens[..., CH_LOG_GK])
        Ak = torch.pow(10.0, tokens[..., CH_LOG_AK])
        ln_gkA = (tokens[..., CH_LOG_GK] + tokens[..., CH_LOG_AK]) * LN10
        z = tokens[..., CH_ION].round().clamp(0.0, 1.0)
        z_int = z.long()
        area = tokens[..., CH_AREA]

        Z = tokens[..., CH_Z].round().long().clamp(0, self.z_to_elem.numel() - 1)
        e_idx = self.z_to_elem[Z]                                           # [B, L], -1 = not target
        is_target = e_idx >= 0
        e_safe = torch.where(is_target, e_idx, torch.zeros_like(e_idx))
        valid = (
            (fit_valid.to(f64) > 0.5) & is_target
            & torch.isfinite(area) & (area > 0.0) & (area > self.min_area) & (wl > 0.0)
        )
        w = torch.where(valid, weights.to(f64), torch.zeros_like(area))
        if self.use_isolation and self.isolation_score is not None:
            w = w * self.isolation_score[None, :L]
        w = torch.where(torch.isfinite(w), w.clamp(min=0.0), torch.zeros_like(w))
        onehot = F.one_hot(e_safe, E).to(f64) * is_target[..., None].to(f64)  # [B, L, E]

        iso_line = (self.isolation_score[None, :].expand(B, L)
                    if getattr(self, "isolation_score", None) is not None
                    else torch.ones(B, L, dtype=f64, device=dev))
        r2_line = (tokens[:, :, CH_R2].to(f64) if tokens.shape[2] > CH_R2
                   else torch.ones(B, L, dtype=f64, device=dev))
        clean_single = ((iso_line >= self.single_line_min_isolation)
                        & (r2_line >= self.single_line_r2_min))

        def gate(w):
            """Zero the lines of elements with fewer than ``min_lines`` weighted lines
            (masks are detached; gradients flow through the surviving weights).

            An element left with a single line must also carry a clean one
            (rule A), otherwise a blend lets it claim matrix mass."""
            used = w > 0.0
            n = (onehot * used[..., None].to(f64)).sum(dim=1)                 # [B, E]
            if self.single_line_min_isolation > 0.0 or self.single_line_r2_min > 0.0:
                lone = torch.gather((n == 1).to(f64), 1, e_safe) > 0.5
                w = torch.where(lone & used & ~clean_single, torch.zeros_like(w), w)
                used = w > 0.0
                n = (onehot * used[..., None].to(f64)).sum(dim=1)
            ok = (n >= self.min_lines).to(f64)
            w = w * torch.gather(ok, 1, e_safe)
            used = w > 0.0
            n = (onehot * used[..., None].to(f64)).sum(dim=1).long()
            return w, used, n, n > 0

        w, used, n_lines_used, has_lines = gate(w)

        E_ion_line = self.E_ion[e_safe]
        mass_line = self.mass_amu[e_safe]
        Eeff = Ek + z * E_ion_line
        A = torch.cat([onehot, -Eeff[..., None], -z[..., None]], dim=-1)     # [B, L, E+2]

        one = torch.ones_like(area)
        area_safe = torch.where(valid, area, one)
        wl_safe = torch.where(wl > 0.0, wl, one)
        y_base = torch.log(area_safe * wl_safe) - ln_gkA

        # seed composition
        c0_given = C0 is not None
        if not c0_given:
            C0 = torch.full((B, E), 1.0 / E, dtype=f64, device=dev)
        C0 = C0.to(f64)
        x0 = C0 / self.mass_amu[None, :]
        x0 = x0 / x0.sum(dim=1, keepdim=True).clamp(min=eps)

        # initial guesses
        if T0 is None:
            T0 = torch.full((B,), 10000.0, dtype=f64, device=dev)
        if log10_Ne0 is None:
            log10_Ne0 = torch.full((B,), 17.0, dtype=f64, device=dev)
        if log10_Nl0 is None:
            log10_Nl0 = torch.full((B,), 16.0, dtype=f64, device=dev)
        T0 = self._clamp_T(T0.to(f64))
        beta0 = 1.0 / (KB_EV * T0)
        eta0 = log10_Ne0.to(f64).clamp(LOG10_NE_MIN, LOG10_NE_MAX) * LN10
        Nl = torch.pow(10.0, log10_Nl0.to(f64))

        prior_diag = torch.zeros(E + 2, dtype=f64, device=dev)
        prior_diag[E] = self.prior_T
        prior_diag[E + 1] = self.prior_Ne
        reg = torch.diag(prior_diag + self.ridge)[None]                      # [1, E+2, E+2]

        def closure(T, lnNe, q, has_lines):
            has_lines_f = has_lines.to(f64)
            U_T = self._partition_functions(T)                               # [B, E, 2]
            S10 = self._saha_ratio(T, lnNe, U_T)                             # [B, E]
            x_tilde = torch.where(has_lines, U_T[:, :, 0] * torch.exp(q) * (1.0 + S10),
                                  torch.zeros_like(q))
            if c0_given and self.seed_in_closure:
                scale = ((x0 * has_lines_f).sum(dim=1).clamp(min=eps)
                         / x_tilde.sum(dim=1).clamp(min=eps))[:, None]
                x = torch.where(has_lines, x_tilde * scale, x0)
            elif c0_given:
                # only identified elements are summed; a row with nothing
                # identified at all keeps the seed so the output stays finite
                x = torch.where(has_lines.any(dim=1, keepdim=True), x_tilde, x0)
            else:
                x = x_tilde
            return x / x.sum(dim=1, keepdim=True).clamp(min=eps), S10, U_T

        def optical_depth(T, x, S10, U_T):
            Tl = T[:, None]
            sigma = wl * torch.sqrt(_KB * Tl / (mass_line * AMU_G * C_CGS ** 2))   # doppler_sigma_nm
            a = self.gamma_nm / sigma
            phi_peak = torch.special.erfcx(a / _SQRT2) / (sigma * _SQRT2PI)       # voigt_peak (per nm)
            U_line = U_T[torch.arange(B, device=dev)[:, None], e_safe, z_int]       # [B, L]
            wl_cm = wl * NM_TO_CM
            kt = (wl_cm ** 4 * _KT_PREFACTOR * Ak * gk
                  * torch.exp(-Ei * _EV_OVER_KB / Tl)
                  * (1.0 - torch.exp(-(Ek - Ei) * _EV_OVER_KB / Tl)) / U_line)     # line_absorption_int
            r_I = 1.0 / (1.0 + S10)
            r_II = S10 / (1.0 + S10)
            r_line = torch.where(z > 0.5, torch.gather(r_II, 1, e_safe), torch.gather(r_I, 1, e_safe))
            x_line = torch.gather(x, 1, e_safe)
            tau0 = kt * x_line * r_line * Nl[:, None] * phi_peak / NM_TO_CM
            return torch.where(is_target, tau0, torch.zeros_like(tau0)), a

        def sa_log_factor(tau0, a):
            f = self._curve_of_growth(tau0, a).clamp(min=eps)
            return torch.where(valid, torch.log(f), torch.zeros_like(f))

        T = T0
        lnNe = eta0
        ln_f = torch.zeros_like(area)
        q = torch.zeros(B, E, dtype=f64, device=dev)
        x = torch.zeros(B, E, dtype=f64, device=dev)
        tau0 = torch.zeros_like(area)
        resid = torch.zeros_like(area)

        if self.sa_correction and self.sa_seed_init and c0_given:
            U_T = self._partition_functions(T)
            S10 = self._saha_ratio(T, lnNe, U_T)
            tau0, a = optical_depth(T, x0, S10, U_T)
            ln_f = sa_log_factor(tau0, a)

        for it in range(self.n_iter):
            y = y_base - ln_f - z * self._ln_saha_thermal_factor(T)[:, None]
            Aw = A * w[..., None]
            M = torch.einsum("bli,blj->bij", Aw, A) + reg
            b = torch.einsum("bli,bl->bi", Aw, y)
            q_bias = torch.zeros(B, E, dtype=f64, device=dev)
            if self.single_line_prior > 0.0 and c0_given:
                # rule B: a single-line element's intercept is pulled toward the
                # seed, anchored on the elements several lines already determine
                lone = (n_lines_used == 1)
                if bool(lone.any()):
                    U_T = self._partition_functions(T)
                    S10 = self._saha_ratio(T, lnNe, U_T)
                    q_seed = (torch.log(x0.clamp(min=eps))
                              - torch.log((U_T[:, :, 0] * (1.0 + S10)).clamp(min=eps)))
                    anchor = (n_lines_used >= 2)
                    gap = torch.where(anchor, q - q_seed, torch.full_like(q, float("nan")))
                    c = _nanmedian_lower(gap, dim=1)[:, None]
                    c = torch.where(torch.isfinite(c), c, torch.zeros_like(c))
                    pw = torch.where(lone, self.single_line_prior, 0.0)
                    M = M + torch.diag_embed(torch.cat(
                        [pw, torch.zeros(B, 2, dtype=f64, device=dev)], dim=1))
                    q_bias = pw * (q_seed + c)
            b = b + torch.cat([q_bias,
                               (self.prior_T * beta0)[:, None], (self.prior_Ne * eta0)[:, None]], dim=1)
            theta = torch.linalg.solve(M, b[..., None])[..., 0]              # [B, E+2]
            q = theta[:, :E]
            beta = theta[:, E]
            eta = theta[:, E + 1]
            T_raw = torch.where(beta > 0, 1.0 / (KB_EV * beta.clamp(min=1e-12)),
                                torch.full_like(beta, self.T_max))
            T = self._clamp_T(T_raw)
            lnNe = eta.clamp(LOG10_NE_MIN * LN10, LOG10_NE_MAX * LN10)
            resid = torch.where(valid, y - torch.einsum("bli,bi->bl", A, theta), torch.zeros_like(y))

            # robust line rejection (detached masks), not after the last solve
            if self.reject_sigma > 0 and it < self.n_iter - 1:
                with torch.no_grad():
                    r = torch.where(used, resid, torch.full_like(resid, float("nan")))
                    n_used = used.sum(dim=1)
                    med = _nanmedian_lower(r, dim=1)                                # [B]
                    mad = _nanmedian_lower((r - med[:, None]).abs(), dim=1)
                    sig = torch.clamp(1.4826 * mad, min=self.reject_floor)
                    sig = torch.where(torch.isfinite(sig), sig, torch.full_like(sig, self.reject_floor))
                    bad = used & ((resid - med[:, None]).abs() > self.reject_sigma * sig[:, None])
                    bad = bad & (n_used >= 4)[:, None]
                if bool(bad.any()):
                    w, used, n_lines_used, has_lines = gate(w * (~bad).to(f64))

            x, S10, U_T = closure(T, lnNe, q, has_lines)
            tau0, a = optical_depth(T, x, S10, U_T)
            if self.sa_correction:
                ln_f = sa_log_factor(tau0, a)

        mass = x * self.mass_amu[None, :]
        mass = mass / mass.sum(dim=1, keepdim=True).clamp(min=eps)
        intercepts = torch.where(has_lines, q, torch.full_like(q, float("nan")))
        censored = (mass < self.lod[None, :]) | (~has_lines & (C0 < self.lod[None, :]))

        return {
            "concentrations": mass,
            "number_fractions": x,
            "T": T,
            "log10_Ne": lnNe / LN10,
            "intercepts": intercepts,
            "tau0": tau0,
            "resid": resid,
            "n_lines_used": n_lines_used,
            "censored": censored,
            "used_mask": used,
            "has_lines": has_lines,
        }
