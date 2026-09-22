"""
Per-session wavelength-axis correction derived from the Voigt fits.

The measured spectra share one wavelength axis, but each measurement session
(the instrument token of ``unique_id``) has its own calibration: the fitted
centres of well-isolated lines sit a stable 0.02–0.09 nm off the database
wavelengths, the offset varies smoothly with wavelength inside each
spectrometer channel and jumps at the channel boundary (~250 nm on the
Chameleon/OptiCal axis).  With 0.05 nm wide lines that is up to two line
widths, so any model that places lines at their database wavelengths
(``cf.deconv``) fits the wrong pixels.

``session_axis_shift`` fits, per session and per channel, a robust polynomial
``delta(lambda)`` through the per-line median offsets (fitted − database) of
the directly fitted lines, and returns the shift sampled on the axis.  The
corrected axis is ``wl - delta(wl)``: a database line then sits where its peak
was measured.  Residuals after the correction are ~0.002 nm (median) against
0.03 nm before; the ~10 % of lines left more than 0.02 nm off are individual
database/blend problems, not axis errors.
"""

from __future__ import annotations

import numpy as np

CH_WL, CH_AREA, CH_R2, CH_DLAM = 0, 9, 11, 12


def robust_polyfit(x: np.ndarray, y: np.ndarray, deg: int, iters: int = 3,
                   floor: float = 0.005) -> np.ndarray:
    """Polynomial fit with 3-sigma (MAD) re-weighting; returns coefficients."""
    keep = np.ones(x.size, dtype=bool)
    c = np.polyfit(x, y, deg)
    for _ in range(iters):
        c = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(c, x)
        s = max(1.4826 * float(np.median(np.abs(r[keep]))), floor)
        keep = np.abs(r) <= 3.0 * s
        if keep.sum() < deg + 2:
            break
    return c


def sessions_from_unique_id(unique_id) -> np.ndarray:
    """``{sample}_{INSTRUMENT}_R{run}`` → the instrument/session token."""
    return np.array([str(u).rsplit("_", 2)[1] if str(u).count("_") >= 2 else str(u)
                     for u in unique_id])


def session_axis_shift(
    tokens: np.ndarray,
    fit_valid: np.ndarray,
    sessions: np.ndarray,
    axis: np.ndarray,
    isolation: np.ndarray | None = None,
    min_isolation: float = 0.3,
    min_r2: float = 0.9,
    max_abs_delta: float = 0.1,
    min_shots: int = 5,
    break_nm: float = 250.0,
    deg: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-session shift curves sampled on ``axis``.

    Args:
        tokens: ``[n, L, >=13]`` line tokens (channels 0, 9, 11, 12 used)
        fit_valid: ``[n, L]``
        sessions: ``[n]`` session label per spectrum
        axis: ``[n_px]`` the shared wavelength axis
        isolation: ``[L]`` isolation score (lines below ``min_isolation`` are
            not used to estimate the curve)
        break_nm: channel boundary; each side gets its own polynomial
        deg: polynomial degree per channel

    Returns:
        shift ``[n_sessions, n_px]`` (delta = fitted − database, nm),
        session index ``[n]`` per spectrum, and a stats dict.
    """
    tokens = np.asarray(tokens)
    n, L = tokens.shape[:2]
    wl = tokens[0, :, CH_WL]
    dl = tokens[:, :, CH_DLAM]
    good = ((np.asarray(fit_valid) > 0.5) & (tokens[:, :, CH_R2] >= min_r2)
            & (tokens[:, :, CH_AREA] > 0) & (np.abs(dl) < max_abs_delta))
    if isolation is not None:
        good &= (np.asarray(isolation) >= min_isolation)[None, :]

    labels, sess_idx = np.unique(np.asarray(sessions), return_inverse=True)
    shift = np.zeros((labels.size, axis.size), dtype=np.float64)
    x_axis = (axis - 280.0) / 140.0
    segs = [axis < break_nm, axis >= break_nm]
    resid = []
    n_lines = []
    # global fallback curve (all sessions pooled)
    def _curve(rows):
        med, lam = [], []
        for l in range(L):
            m = rows[good[rows, l]]
            if m.size >= min_shots:
                med.append(float(np.median(dl[m, l]))); lam.append(float(wl[l]))
        return np.asarray(lam), np.asarray(med)

    lam_g, med_g = _curve(np.arange(n))
    for k in range(labels.size):
        rows = np.where(sess_idx == k)[0]
        lam, med = _curve(rows)
        n_lines.append(lam.size)
        for seg_axis, lo, hi in ((segs[0], -np.inf, break_nm), (segs[1], break_nm, np.inf)):
            sel = (lam >= lo) & (lam < hi)
            if sel.sum() >= deg + 4:
                x, y = (lam[sel] - 280.0) / 140.0, med[sel]
            else:                                   # too few lines: pooled curve
                sel_g = (lam_g >= lo) & (lam_g < hi)
                x, y = (lam_g[sel_g] - 280.0) / 140.0, med_g[sel_g]
                if x.size < deg + 4:
                    continue
            c = robust_polyfit(x, y, deg)
            shift[k, seg_axis] = np.polyval(c, x_axis[seg_axis])
            if sel.sum() >= deg + 4:
                resid.append(med[sel] - np.polyval(c, (lam[sel] - 280.0) / 140.0))
    r = np.concatenate(resid) if resid else np.zeros(0)
    stats = dict(n_sessions=int(labels.size), lines_per_session=float(np.median(n_lines)) if n_lines else 0.0,
                 resid_median_nm=float(np.median(np.abs(r))) if r.size else float("nan"),
                 resid_p90_nm=float(np.percentile(np.abs(r), 90)) if r.size else float("nan"),
                 offset_before_median_nm=float(np.median(np.abs(dl[good]))) if good.any() else float("nan"))
    return shift, sess_idx, stats
