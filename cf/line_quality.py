"""
Per-line quality statistics and weighting rules for the CF Saha–Boltzmann solve.

Everything here works on the per-line residuals of the solve itself, so it
needs no reference values: a line that is wrong (bad gA, unlisted interferer,
self-absorption, a deconvolution that split the peak badly) sits off the
Boltzmann line through the other lines of its element in *every* spectrum, and
that is measurable as a per-line bias; a line that is merely noisy has a large
per-line scatter.  The functions return numpy arrays sized like the token cache
so they plug into ``cf.solver_np`` / ``cf.layer`` as ``weights``.

Vocabulary (all masks are ``[n_spectra, n_lines]`` bool):

    direct    the Voigt fit is valid, r² ≥ r2_min, isolation ≥ iso_min, area > 0
    rescued   a line that is not direct but has a deconvolved area
              (``scripts/deconvolve_lines.py``); its token area is replaced

``line_consistency`` is the diagnostic; ``rule_weights`` the rule-based
selection; ``irls_weights`` the per-spectrum robust reweighting.  The trained
alternative (per-line weights fitted to certified samples, grouped CV) lives in
``scripts/select_cf_lines.py`` because it needs the torch layer and the
reference table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CH_WL, CH_EK, CH_AREA, CH_R2 = 0, 2, 9, 11


@dataclass
class LineMasks:
    direct: np.ndarray            # [n, L] bool
    rescued: np.ndarray           # [n, L] bool
    blend: np.ndarray             # [n, L] float, 1 where unknown
    window_r2: np.ndarray         # [n, L] float, 0 where unknown


def build_masks(tokens: np.ndarray, fit_valid: np.ndarray, isolation: np.ndarray,
                deconv_area: np.ndarray | None = None, deconv_blend: np.ndarray | None = None,
                deconv_r2: np.ndarray | None = None, rescue_elements: np.ndarray | None = None,
                iso_min: float = 0.3, r2_min: float = 0.9, max_blend: float = 0.8
                ) -> tuple[np.ndarray, np.ndarray, LineMasks]:
    """Direct/rescued masks and the token arrays with the rescued areas filled in.

    Args:
        rescue_elements: ``[L]`` bool, lines eligible for rescue (e.g. the Fe
            lines); ``None`` = every line with a deconvolved area

    Returns: ``(tokens_aug, fit_valid_aug, masks)``.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    fv = np.asarray(fit_valid, dtype=np.float64)
    direct = ((fv > 0.5) & (tokens[:, :, CH_R2] >= r2_min)
              & (np.asarray(isolation) >= iso_min)[None, :] & (tokens[:, :, CH_AREA] > 0))
    n, L = direct.shape
    if deconv_area is None:
        rescued = np.zeros_like(direct)
        blend = np.ones((n, L)); wr2 = np.zeros((n, L))
    else:
        blend = np.nan_to_num(np.asarray(deconv_blend, dtype=np.float64), nan=1.0)
        wr2 = (np.nan_to_num(np.asarray(deconv_r2, dtype=np.float64), nan=0.0)
               if deconv_r2 is not None else np.zeros((n, L)))
        rescued = ~direct & (np.asarray(deconv_area) > 0) & (blend <= max_blend)
        if rescue_elements is not None:
            rescued &= np.asarray(rescue_elements, dtype=bool)[None, :]
    tok_aug = tokens.copy(); fv_aug = fv.copy()
    if rescued.any():
        tok_aug[rescued, CH_AREA] = np.asarray(deconv_area, dtype=np.float64)[rescued]
        fv_aug = np.where(rescued, 1.0, fv_aug)
    return tok_aug, fv_aug, LineMasks(direct=direct, rescued=rescued, blend=blend, window_r2=wr2)


def line_consistency(resid: np.ndarray, mask: np.ndarray, sample_ids: np.ndarray,
                     tokens: np.ndarray, symbols: np.ndarray, min_count: int = 30
                     ) -> pd.DataFrame:
    """Per-line Boltzmann self-consistency over many spectra.

    ``resid`` is the solver's ``y − model`` at the final state (``[n, L]``);
    for a line with zero weight it is the distance of that line's point from
    the fit through the *weighted* lines of its element, so passing the
    residuals of a direct-lines-only solve and the ``rescued`` mask measures
    the deconvolved points against the clean fit.

    Columns: line, el, wl, Ek, n, bias (median residual), scatter (1.4826·MAD),
    within (median over samples of the residual std among a sample's shots).
    """
    rows = []
    wl = tokens[0, :, CH_WL]; Ek = tokens[0, :, CH_EK]
    sample_ids = np.asarray(sample_ids)
    for l in range(resid.shape[1]):
        m = mask[:, l]
        if m.sum() < min_count:
            continue
        r = resid[m, l]
        med = float(np.median(r)); mad = 1.4826 * float(np.median(np.abs(r - med)))
        g = pd.Series(r).groupby(sample_ids[m]).std()
        rows.append(dict(line=l, el=str(symbols[l]), wl=float(wl[l]), Ek=float(Ek[l]), n=int(m.sum()),
                         bias=med, scatter=mad, within=float(np.nanmedian(g))))
    return pd.DataFrame(rows, columns=["line", "el", "wl", "Ek", "n", "bias", "scatter", "within"])


def estimator_agreement(area_a: np.ndarray, area_b: np.ndarray, mask: np.ndarray,
                        groups: np.ndarray | None = None) -> pd.DataFrame:
    """ln(area_a / area_b) where both exist: the check that a deconvolution
    reproduces the Voigt area on isolated lines (median ≈ 0, small sigma)."""
    both = mask & (area_a > 0) & (area_b > 0)
    out = []
    keys = [None] if groups is None else [None, *np.unique(groups)]
    for k in keys:
        m = both if k is None else both & (np.asarray(groups) == k)[None, :]
        if m.sum() < 20:
            continue
        v = np.log(area_a[m] / area_b[m]); med = float(np.median(v))
        out.append(dict(group="all" if k is None else str(k), n=int(m.sum()), ln_ratio=med,
                        ratio=float(np.exp(med)), sigma=1.4826 * float(np.median(np.abs(v - med)))))
    return pd.DataFrame(out)


def per_line_stats(table: pd.DataFrame, n_lines: int) -> tuple[np.ndarray, np.ndarray]:
    """``bias[L]`` (0 where unknown) and ``scatter[L]`` (inf where unknown)."""
    bias = np.zeros(n_lines); scatter = np.full(n_lines, np.inf)
    if len(table):
        bias[table["line"].to_numpy()] = table["bias"].to_numpy()
        scatter[table["line"].to_numpy()] = table["scatter"].to_numpy()
    return bias, scatter


def rule_weights(masks: LineMasks, rule: str = "blend", *, max_blend: float = 0.8,
                 min_window_r2: float = 0.0, bias: np.ndarray | None = None,
                 scatter: np.ndarray | None = None, max_abs_bias: float = np.inf,
                 max_scatter: float = np.inf, s0: float = 0.3,
                 direct_weight: float = 1.0) -> np.ndarray:
    """Per-line weights ``[n, L]``: direct lines get ``direct_weight``; rescued
    lines get a rule-dependent weight, zero when they fail the filters.

    rules:
        none      rescued lines are not used
        blend     w = 1 − blend  (blend ≤ max_blend, window r² ≥ min_window_r2)
        filter    as ``blend`` plus |bias| ≤ max_abs_bias and scatter ≤ max_scatter
        invvar    (1 − blend) · s0² / (scatter² + s0²)  — a weighted Boltzmann plot
    """
    n, L = masks.direct.shape
    W = masks.direct.astype(np.float64) * direct_weight
    if rule == "none":
        return W
    ok = masks.rescued & (masks.blend <= max_blend) & (masks.window_r2 >= min_window_r2)
    w = 1.0 - masks.blend
    if rule in ("filter", "invvar"):
        b = np.zeros(L) if bias is None else np.asarray(bias)
        s = np.full(L, np.inf) if scatter is None else np.asarray(scatter)
        if rule == "filter":
            ok &= ((np.abs(b) <= max_abs_bias) & (s <= max_scatter))[None, :]
        else:
            w = w * (s0 ** 2 / (s ** 2 + s0 ** 2))[None, :]
    elif rule != "blend":
        raise ValueError(f"unknown rule {rule!r}")
    return W + np.where(ok, w, 0.0)


def bias_corrected_area(tokens_aug: np.ndarray, masks: LineMasks, bias: np.ndarray) -> np.ndarray:
    """Divide the rescued areas by exp(bias): a per-line efficiency factor from
    self-consistency (no reference values).  Returns a new token array."""
    tok = tokens_aug.copy()
    f = np.exp(-np.asarray(bias))[None, :].repeat(tok.shape[0], 0)
    tok[masks.rescued, CH_AREA] = tok[masks.rescued, CH_AREA] * f[masks.rescued]
    return tok


def irls_weights(W: np.ndarray, resid: np.ndarray, c: float = 0.3,
                 direct: np.ndarray | None = None) -> np.ndarray:
    """One IRLS step with a Cauchy influence function: w ← w / (1 + (r/c)²).
    With ``direct`` given, the direct lines keep their weight."""
    W2 = W / (1.0 + (resid / c) ** 2)
    if direct is not None:
        W2 = np.where(direct, W, W2)
    return W2
