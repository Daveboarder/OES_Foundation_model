"""
Calibrate the physics_version-2 generator (`data/two_zone_pipeline.py`)
against measured PURE KFE (pure iron) spark spectra.

(i)  Instrument FWHM: Gaussian (+ linear baseline) fits to isolated, weak,
     optically thin Fe lines of the measured spectrum; the median FWHM is the
     recommended `generation.instrument.fwhm_nm` (it is an upper bound: the
     intrinsic Stark/Doppler width is folded in).
(ii) Path-length calibration: the saturation statistic
        S1 = area(top-5 strong Fe I resonance lines, E_i < 0.2 eV)
           / area(top-5 weak, high-E_k Fe I lines)
     is measured on the PURE KFE spectra and on synthetic one-zone pure-Fe
     spectra at a few (Te, Ne) with the inner path length l on a log grid;
     the l where they match is reported per (Te, Ne).  A second statistic
        S2 = area(top-5 resonance lines) / area(top-5 strong lines with
             0.8 < E_i < 1.6 eV)
     isolates the cold-shell absorption (only E_i ~ 0 lines feel it) and is
     used the same way for l_outer on two-zone spectra (Te2 = 0.35 Te1,
     Ne2 = 0.01 Ne1, l_inner fixed at its one-zone match).

Both statistics are ratios of areas measured with the same routine on the
measured and on the synthetic spectrum (interpolated onto the VASKUT axis,
instrument-convolved), so scale, sampling and baseline effects cancel to
first order.  The ratio S1 is also a function of temperature (Boltzmann
factor between E_k ~ 5 eV and ~ 6.3 eV), which is why several (Te, Ne) are
reported: pick the row closest to the expected spark temperature.

Results: Outputs/calibrate_generator_<ts>.json and .png, and a printed
YAML snippet with the recommended values.  This script never edits configs.

Usage:
    uv run python scripts/calibrate_generator.py
    uv run python scripts/calibrate_generator.py --file "REMUS-9951602/FEGLFE/PURE KFE.json" --max_files 20
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import plasma_physics as pp
from data.libs_pipeline import load_wavelength
from data.measured_pipeline import find_first_valid_run, load_spectrum_from_json
from data.two_zone_pipeline import _number_density_rows, generate_zone_sample_table, synthesise_spectrum
from external_data.Context.readData import json_from_file

T_REF, NE_REF = 10000.0, 1e17          # reference plasma for line ranking
FE_MASS = 55.845
DEFAULT_FILE = "REMUS-9951602/FEGLFE/PURE KFE.json"
ONE_ZONE_STATES = [(8000.0, 1e17), (10000.0, 3e17), (12000.0, 1e18), (15000.0, 1.79e18)]
L_INNER_GRID = np.logspace(-6, 0, 25)      # cm
L_OUTER_GRID = np.logspace(-8, -1, 29)     # cm
TE2_RATIO, NE2_RATIO = 0.35, 1e-2
GAMMA_NM = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Spectrometer axis helpers (two CCD segments overlap; keep one monotonic axis)
# ─────────────────────────────────────────────────────────────────────────────
def monotonic_axis(wavelength: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Index selection giving a strictly increasing axis: CCD 1 in full, CCD 2
    only beyond CCD 1's end. Returns (indices, (overlap_lo, overlap_hi))."""
    d = np.diff(wavelength)
    breaks = np.nonzero(d < 0)[0]
    if breaks.size == 0:
        return np.arange(wavelength.size), (np.nan, np.nan)
    n1 = int(breaks[0]) + 1
    lo, hi = float(wavelength[n1]), float(wavelength[n1 - 1])
    idx2 = np.nonzero(wavelength[n1:] > hi)[0] + n1
    return np.concatenate([np.arange(n1), idx2]), (lo, hi)


def line_area(wl: np.ndarray, s: np.ndarray, centre: float, fwhm: float,
              search_nm: float = 0.05) -> tuple[float, float, float, float]:
    """Baseline-subtracted area of the line nearest ``centre``.

    Baseline = 10th percentile in +-0.5 nm, peak searched in +-search_nm,
    area = trapezoid of (s - baseline) over peak +- 2 fwhm.
    Returns (area, peak_height, noise, peak_position)."""
    bw = (wl > centre - 0.5) & (wl < centre + 0.5)
    if bw.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan
    base = float(np.percentile(s[bw], 10))
    low = np.sort(s[bw])[: max(4, int(0.3 * bw.sum()))]
    noise = float(np.std(low)) if low.size > 2 else np.nan
    pw = (wl > centre - search_nm) & (wl < centre + search_nm)
    if not pw.any():
        return np.nan, np.nan, np.nan, np.nan
    k = np.nonzero(pw)[0][np.argmax(s[pw])]
    pos = float(wl[k])
    aw = (wl > pos - 2 * fwhm) & (wl < pos + 2 * fwhm)
    area = float(np.trapezoid(s[aw] - base, wl[aw]))
    return area, float(s[k] - base), noise, pos


def _gauss_lin(x, a, mu, sig, b, c):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b + c * (x - mu)


def fit_gaussian(wl: np.ndarray, s: np.ndarray, centre: float, half: float = 0.15) -> dict | None:
    w = (wl > centre - half) & (wl < centre + half)
    if w.sum() < 7:
        return None
    x, y = wl[w], s[w]
    k = int(np.argmax(y))
    b0 = float(np.percentile(y, 10))
    p0 = [max(y[k] - b0, 1e-9), float(x[k]), 0.02, b0, 0.0]
    try:
        popt, _ = curve_fit(
            _gauss_lin, x, y, p0=p0,
            bounds=([0, centre - half, 0.004, -np.inf, -np.inf], [np.inf, centre + half, 0.12, np.inf, np.inf]),
            maxfev=4000,
        )
    except Exception:
        return None
    yhat = _gauss_lin(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"centre": centre, "mu": float(popt[1]), "fwhm": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * popt[2]),
            "amplitude": float(popt[0]), "r2": r2, "n_points": int(w.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# Line selection from the DB (pure Fe)
# ─────────────────────────────────────────────────────────────────────────────
class FeLines:
    def __init__(self, db_path: str, axis_lo: float, axis_hi: float, overlap: tuple[float, float]):
        ls = pp.line_set_for_element("Fe", T_REF, NE_REF, db_path)
        self.wl = ls.wl_nm
        self.is_I = ls.is_I
        self.Ei, self.Ek, self.Ak = ls.Ei, ls.Ek, ls.Ak
        self.I = ls.eps_per_n
        self.i_max = float(np.max(self.I[self.is_I & (self.Ei < 0.2)]))
        in_axis = (self.wl > axis_lo + 0.5) & (self.wl < axis_hi - 0.5)
        if np.isfinite(overlap[0]):
            in_axis &= ~((self.wl > overlap[0] - 0.3) & (self.wl < overlap[1] + 0.3))
        self.usable = in_axis

    def isolated(self, window: float, frac: float) -> np.ndarray:
        """No other Fe line within +-window whose reference intensity exceeds
        ``frac`` of the candidate's."""
        order = np.argsort(self.wl)
        wl_s, I_s = self.wl[order], self.I[order]
        out = np.ones(self.wl.size, dtype=bool)
        for k in range(self.wl.size):
            lo = np.searchsorted(wl_s, self.wl[k] - window)
            hi = np.searchsorted(wl_s, self.wl[k] + window)
            neigh = I_s[lo:hi]
            # exclude the line itself (largest exact match on wavelength)
            self_mask = wl_s[lo:hi] == self.wl[k]
            if np.any(neigh[~self_mask] > frac * self.I[k]):
                out[k] = False
        return out

    def pick(self, mask: np.ndarray, n: int) -> np.ndarray:
        cand = np.nonzero(mask & self.usable)[0]
        return cand[np.argsort(-self.I[cand])][:n]


def select_line_sets(fe: FeLines, wl_m: np.ndarray, s_m: np.ndarray, fwhm: float, snr_min: float = 10.0) -> dict:
    """Resonance / weak high-E_k / intermediate-E_i sets, each restricted to
    lines that show a clear peak in the measured spectrum."""
    iso_tight = fe.isolated(0.1, 0.05)
    detected = np.zeros(fe.wl.size, dtype=bool)
    for k in np.nonzero(fe.usable & iso_tight)[0]:
        _, h, noise, _ = line_area(wl_m, s_m, fe.wl[k], fwhm)
        detected[k] = np.isfinite(h) and noise > 0 and h > snr_min * noise
    rel = fe.I / fe.i_max
    sets = {
        "resonance": fe.pick(fe.is_I & (fe.Ei < 0.2) & iso_tight & detected, 5),
        "weak_high_Ek": fe.pick(fe.is_I & (fe.Ei > 2.5) & (rel > 1e-3) & (rel < 5e-2) & iso_tight & detected, 5),
        "intermediate_Ei": fe.pick(fe.is_I & (fe.Ei > 0.8) & (fe.Ei < 1.6) & iso_tight & detected, 5),
    }
    return sets


def statistics(wl: np.ndarray, s: np.ndarray, fe: FeLines, sets: dict, fwhm: float) -> dict[str, float]:
    areas = {name: np.array([line_area(wl, s, fe.wl[k], fwhm)[0] for k in idx]) for name, idx in sets.items()}
    a_res, a_weak, a_mid = (np.nansum(areas[n]) for n in ("resonance", "weak_high_Ek", "intermediate_Ei"))
    return {
        "S1": float(a_res / a_weak) if a_weak > 0 else np.nan,
        "S2": float(a_res / a_mid) if a_mid > 0 else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic pure-Fe spectra
# ─────────────────────────────────────────────────────────────────────────────
def fe_row(Te1, Ne1, l_inner, db_path, Te2=None, Ne2=None, l_outer=0.0, nd_max=None) -> dict:
    """Contract-C1 row for pure Fe (quasi-neutral N per zone, optional cap)."""
    N1 = float(_number_density_rows(["Fe"], np.array([[1.0]]), np.array([Te1]), np.array([Ne1]), db_path, nd_max)[0])
    row = {"plasma_model": "one_zone", "Te1": Te1, "Ne1": Ne1, "Te2": Te1, "Ne2": Ne1,
           "l_inner": l_inner, "l_outer": 0.0, "N1": N1, "N2": N1, "gamma_stark1": GAMMA_NM, "gamma_stark2": GAMMA_NM}
    if Te2 is not None and l_outer > 0:
        N2 = float(_number_density_rows(["Fe"], np.array([[1.0]]), np.array([Te2]), np.array([Ne2]), db_path, nd_max)[0])
        row.update({"plasma_model": "two_zone", "Te2": Te2, "Ne2": Ne2, "l_outer": l_outer, "N2": N2})
    return row


def crossings(l_grid: np.ndarray, values: np.ndarray, target: float) -> list[float]:
    """l where log(values) crosses log(target) (linear interpolation in log-log)."""
    x = np.log10(l_grid)
    y = np.log10(values) - np.log10(target)
    out = []
    for i in range(len(x) - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        if y[i] == 0:
            out.append(float(l_grid[i]))
        elif y[i] * y[i + 1] < 0:
            t = y[i] / (y[i] - y[i + 1])
            out.append(float(10 ** (x[i] + t * (x[i + 1] - x[i]))))
    return out


def _round_sig(v: float, n: int = 1) -> float:
    if not np.isfinite(v) or v <= 0:
        return v
    e = np.floor(np.log10(v))
    return float(np.round(v / 10 ** e, n) * 10 ** e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs_data_config", default="config/libs_data_measured.yaml")
    p.add_argument("--generator_config", default="config/libs_data.yaml",
                   help="libs_data YAML whose generation block (zones, number_density_max, fine_step_nm, ...) is used for the synthetic spectra")
    p.add_argument("--file", default=DEFAULT_FILE, help="PURE KFE JSON (relative to measured_json_root) used for line selection and FWHM")
    p.add_argument("--max_files", type=int, default=None, help="cap on PURE KFE files for the measured statistic distribution")
    p.add_argument("--out_dir", default="Outputs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.libs_data_config))
    gen_cfg_file = yaml.safe_load(open(args.generator_config)).get("generation", {})
    nd_max = gen_cfg_file.get("number_density_max")
    nd_max = None if nd_max is None else float(nd_max)
    root = Path(cfg["paths"]["measured_json_root"]).expanduser()
    db_path = str(Path(cfg["paths"]["db"]).resolve())
    wl_json = str(Path(cfg["paths"]["wavelength_json"]).resolve())
    wavelength = load_wavelength(wl_json)
    idx_mono, overlap = monotonic_axis(wavelength)
    wl_m = wavelength[idx_mono]

    files = sorted(glob.glob(str(root / "*" / "*" / "PURE KFE.json")))
    if not files:
        raise SystemExit(f"no PURE KFE.json under {root}")
    main_file = str(root / args.file)
    if main_file not in files:
        print(f"WARN: {args.file} not found; using {files[0]}")
        main_file = files[0]
    if args.max_files:
        files = files[: args.max_files]
    print(f"PURE KFE files: {len(files)} (reference: {Path(main_file).relative_to(root)})")

    def load(path: str) -> np.ndarray:
        an = json_from_file(path)["analysis"]
        run = find_first_valid_run(an)
        return load_spectrum_from_json(path, run, 1, (1, 2), wavelength, normalize=False)[idx_mono]

    s_ref = load(main_file)
    fe = FeLines(db_path, float(wl_m.min()), float(wl_m.max()), overlap)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"timestamp": ts, "reference_file": main_file, "n_files": len(files),
                    "overlap_region_nm": list(overlap)}

    # ---- (i) instrument FWHM --------------------------------------------------
    t0 = time.time()
    rel = fe.I / fe.i_max
    iso_wide = fe.isolated(0.3, 0.05)
    cand = np.nonzero(fe.usable & iso_wide & (rel > 1e-3) & (rel < 5e-2))[0]
    fits = []
    for k in cand:
        _, h, noise, _ = line_area(wl_m, s_ref, fe.wl[k], 0.05)
        if not (np.isfinite(h) and noise > 0 and h > 20 * noise):
            continue
        f = fit_gaussian(wl_m, s_ref, fe.wl[k])
        if f and f["r2"] > 0.98 and abs(f["mu"] - fe.wl[k]) < 0.05:
            f["Ei"], f["Ek"], f["stage"] = float(fe.Ei[k]), float(fe.Ek[k]), "I" if fe.is_I[k] else "II"
            fits.append(f)
    fits.sort(key=lambda f: -f["r2"])
    fits = fits[:20]
    fwhms = np.array([f["fwhm"] for f in fits])
    fwhm_med = float(np.median(fwhms)) if fwhms.size else np.nan
    fwhm_mad = float(np.median(np.abs(fwhms - fwhm_med))) if fwhms.size else np.nan
    result["instrument"] = {"fwhm_median_nm": fwhm_med, "fwhm_mad_nm": fwhm_mad, "n_lines": int(fwhms.size),
                            "fwhm_min_nm": float(fwhms.min()) if fwhms.size else np.nan,
                            "fwhm_max_nm": float(fwhms.max()) if fwhms.size else np.nan,
                            "lines": fits}
    print(f"\n(i) instrument FWHM from {fwhms.size} isolated weak Fe lines ({time.time() - t0:.1f} s): "
          f"median {fwhm_med:.4f} nm, MAD {fwhm_mad:.4f} nm, range [{fwhms.min():.4f}, {fwhms.max():.4f}]")
    for f in fits[:8]:
        print(f"     {f['centre']:.3f} nm  Fe {f['stage']}  Ek {f['Ek']:.2f} eV  fwhm {f['fwhm']:.4f}  r2 {f['r2']:.3f}")
    fwhm_use = fwhm_med if np.isfinite(fwhm_med) else 0.05

    # ---- (ii) measured statistics ---------------------------------------------
    t0 = time.time()
    sets = select_line_sets(fe, wl_m, s_ref, fwhm_use)
    result["line_sets"] = {name: [{"wl": float(fe.wl[k]), "Ei": float(fe.Ei[k]), "Ek": float(fe.Ek[k]),
                                   "Ak": float(fe.Ak[k]), "I_rel": float(fe.I[k] / fe.i_max)} for k in idx]
                           for name, idx in sets.items()}
    print("\n(ii) line sets (DB, vacuum nm):")
    for name, idx in sets.items():
        print(f"     {name:16s}: " + ", ".join(f"{fe.wl[k]:.3f}(Ei {fe.Ei[k]:.2f})" for k in idx))
    if any(len(v) < 3 for v in sets.values()):
        raise SystemExit("fewer than 3 lines in a set — relax the selection")

    meas = []
    for f in files:
        try:
            st = statistics(wl_m, load(f), fe, sets, fwhm_use)
        except Exception as exc:
            print(f"     WARN {f}: {exc}")
            continue
        st["file"] = str(Path(f).relative_to(root))
        meas.append(st)
    S1_m = np.array([m["S1"] for m in meas]); S2_m = np.array([m["S2"] for m in meas])
    q = lambda a: [float(np.nanpercentile(a, 25)), float(np.nanmedian(a)), float(np.nanpercentile(a, 75))]
    S1_q, S2_q = q(S1_m), q(S2_m)
    result["measured"] = {"S1_quartiles": S1_q, "S2_quartiles": S2_q, "per_file": meas}
    print(f"     measured over {len(meas)} files ({time.time() - t0:.1f} s): "
          f"S1 = {S1_q[1]:.3f} [{S1_q[0]:.3f}, {S1_q[2]:.3f}]   S2 = {S2_q[1]:.3f} [{S2_q[0]:.3f}, {S2_q[2]:.3f}]")

    # ---- (ii) synthetic one-zone S1(l) ----------------------------------------
    gen_cfg = {"fine_step_nm": float(gen_cfg_file.get("fine_step_nm", 0.002)),
               "line_window_nm": float(gen_cfg_file.get("line_window_nm", 0.4)),
               "adaptive_window": bool(gen_cfg_file.get("adaptive_window", True)),
               "min_relative_intensity": float(gen_cfg_file.get("min_relative_intensity", 1e-7)),
               "instrument": {**(gen_cfg_file.get("instrument") or {}), "fwhm_nm": fwhm_use}}
    print(f"     synthetic spectra: {gen_cfg}  number_density_max {nd_max}")
    t0 = time.time()
    one_zone = []
    l_inner_matches = []
    for Te, Ne in ONE_ZONE_STATES:
        S1_l, S2_l = [], []
        for l in L_INNER_GRID:
            s = synthesise_spectrum(["Fe"], np.array([1.0]), wavelength, fe_row(Te, Ne, l, db_path, nd_max=nd_max), db_path, gen_cfg)[idx_mono]
            st = statistics(wl_m, s, fe, sets, fwhm_use)
            S1_l.append(st["S1"]); S2_l.append(st["S2"])
        S1_l, S2_l = np.array(S1_l), np.array(S2_l)
        m_med = crossings(L_INNER_GRID, S1_l, S1_q[1])
        m_lo = crossings(L_INNER_GRID, S1_l, S1_q[2])   # larger S -> thinner -> smaller l
        m_hi = crossings(L_INNER_GRID, S1_l, S1_q[0])
        N1 = fe_row(Te, Ne, 1.0, db_path, nd_max=nd_max)["N1"]
        entry = {"Te": Te, "Ne": Ne, "N1": N1, "l_grid": L_INNER_GRID.tolist(), "S1": S1_l.tolist(), "S2": S2_l.tolist(),
                 "S1_thin": float(S1_l[0]), "l_match_median": m_med, "l_match_q25": m_lo, "l_match_q75": m_hi}
        entry["NL1_match_cm-2"] = [N1 * v for v in (m_med + m_lo + m_hi)]
        one_zone.append(entry)
        l_inner_matches.extend(m_med + m_lo + m_hi)
        print(f"     one-zone Te {Te:6.0f} K  Ne {Ne:.2e}  N1 {N1:.2e}: S1 thin {S1_l[0]:.3f}, S1(l) range "
              f"[{np.nanmin(S1_l):.3f}, {np.nanmax(S1_l):.3f}]  l_inner match: median {m_med}, IQR {m_lo} .. {m_hi}")
    result["one_zone"] = one_zone
    print(f"     ({time.time() - t0:.1f} s)")

    # ---- (ii) two-zone S2(l_outer) --------------------------------------------
    t0 = time.time()
    two_zone = []
    l_outer_matches = []
    for entry in one_zone:
        if not entry["l_match_median"]:
            continue
        Te, Ne, l_in = entry["Te"], entry["Ne"], entry["l_match_median"][0]
        Te2, Ne2 = TE2_RATIO * Te, NE2_RATIO * Ne
        S1_l, S2_l = [], []
        for lo in L_OUTER_GRID:
            row = fe_row(Te, Ne, l_in, db_path, Te2, Ne2, lo, nd_max=nd_max)
            s = synthesise_spectrum(["Fe"], np.array([1.0]), wavelength, row, db_path, gen_cfg)[idx_mono]
            st = statistics(wl_m, s, fe, sets, fwhm_use)
            S1_l.append(st["S1"]); S2_l.append(st["S2"])
        S1_l, S2_l = np.array(S1_l), np.array(S2_l)
        N2 = fe_row(Te, Ne, l_in, db_path, Te2, Ne2, 1e-3, nd_max=nd_max)["N2"]
        m_med = crossings(L_OUTER_GRID, S2_l, S2_q[1])
        m_lo = crossings(L_OUTER_GRID, S2_l, S2_q[2])
        m_hi = crossings(L_OUTER_GRID, S2_l, S2_q[0])
        two_zone.append({"Te1": Te, "Ne1": Ne, "l_inner": l_in, "Te2": Te2, "Ne2": Ne2, "N2": N2,
                         "l_outer_grid": L_OUTER_GRID.tolist(), "S1": S1_l.tolist(), "S2": S2_l.tolist(),
                         "S2_no_shell": float(S2_l[0]), "l_outer_match_median": m_med,
                         "l_outer_match_q25": m_lo, "l_outer_match_q75": m_hi,
                         "N2_l_outer_match_cm-2": [N2 * v for v in m_med]})
        l_outer_matches.extend(m_med + m_lo + m_hi)
        print(f"     two-zone Te1 {Te:6.0f} K / l_inner {l_in:.2e} cm, Te2 {Te2:.0f} K Ne2 {Ne2:.1e} N2 {N2:.2e}: "
              f"S2 no shell {S2_l[0]:.3f} -> l_outer match median {m_med}, IQR {m_lo} .. {m_hi}")
    result["two_zone"] = two_zone
    print(f"     ({time.time() - t0:.1f} s)")

    # ---- recommendations --------------------------------------------------------
    # The physical invariant behind each statistic is a column density
    # (N1 * l_inner for the core saturation, N2 * l_outer for the shell), so the
    # l ranges are derived from the matched column densities and the N
    # distribution the generator config actually produces.
    rec = {"instrument": {"profile": "gaussian", "fwhm_nm": float(np.round(fwhm_use, 3))}}
    NL1 = [v for e in one_zone for v in e["NL1_match_cm-2"]]
    NL2 = [v for e in two_zone for v in e["N2_l_outer_match_cm-2"][:1]]
    zones = gen_cfg_file.get("zones") or {}
    draw = generate_zone_sample_table({"Fe": (1.0, 1.0)}, 4000, "FE", "Fe", zones, np.random.default_rng(0),
                                      two_zone_fraction=1.0, number_density="auto", db_path=db_path,
                                      number_density_max=nd_max)
    qN = lambda col: [float(np.percentile(draw[col], q)) for q in (10, 50, 90)]
    N1_q, N2_q = qN("N1"), qN("N2")
    NL1_cfg = qN("N1") and [float(np.percentile(draw["N1"] * draw["l_inner"], q)) for q in (10, 50, 90)]
    NL2_cfg = [float(np.percentile(draw["N2"] * draw["l_outer"], q)) for q in (10, 50, 90)]
    result["config_draw"] = {"zones": zones, "number_density_max": nd_max, "N1_q10_50_90": N1_q, "N2_q10_50_90": N2_q,
                             "N1_l_inner_q10_50_90": NL1_cfg, "N2_l_outer_q10_50_90": NL2_cfg}
    print(f"\nColumn densities (pure Fe): matched core N1*l_inner = "
          f"[{min(NL1):.2e}, {max(NL1):.2e}] cm^-2" if NL1 else "\nNo core match.")
    if NL2:
        print(f"                            matched shell N2*l_outer = [{min(NL2):.2e}, {max(NL2):.2e}] cm^-2")
    print(f"  current {args.generator_config} zones draw (4000 two-zone Fe shots, number_density_max {nd_max}):")
    print(f"     N1  q10/50/90 = {N1_q[0]:.2e} / {N1_q[1]:.2e} / {N1_q[2]:.2e}   N1*l_inner = "
          f"{NL1_cfg[0]:.2e} / {NL1_cfg[1]:.2e} / {NL1_cfg[2]:.2e}")
    print(f"     N2  q10/50/90 = {N2_q[0]:.2e} / {N2_q[1]:.2e} / {N2_q[2]:.2e}   N2*l_outer = "
          f"{NL2_cfg[0]:.2e} / {NL2_cfg[1]:.2e} / {NL2_cfg[2]:.2e}")
    if l_inner_matches:
        rec["l_inner_cm_direct_match"] = [_round_sig(min(l_inner_matches)), _round_sig(max(l_inner_matches))]
    if NL1:
        rec["l_inner_cm"] = [_round_sig(min(NL1) / N1_q[2]), _round_sig(max(NL1) / N1_q[0])]
    if l_outer_matches:
        rec["l_outer_cm_direct_match"] = [_round_sig(min(l_outer_matches)), _round_sig(max(l_outer_matches))]
    if NL2:
        rec["l_outer_cm"] = [_round_sig(min(NL2) / N2_q[2]), _round_sig(max(NL2) / N2_q[0])]
    rec["core_column_density_cm-2"] = [min(NL1), max(NL1)] if NL1 else None
    rec["shell_column_density_cm-2"] = [min(NL2), max(NL2)] if NL2 else None
    result["recommended"] = rec
    print("\nRecommended config values (generation block; copy by hand, this script does not edit configs):")
    print("  (l ranges = matched column density / N quantiles of the config draw; *_direct_match = raw l crossings)")
    print(yaml.safe_dump({"instrument": rec["instrument"],
                          "zones": {k: v for k, v in rec.items() if k.startswith("l_")}}, sort_keys=False).rstrip())
    if not l_inner_matches:
        print("  NOTE: no l_inner crossing found — the measured S1 lies outside the synthetic range for every (Te, Ne).")

    # ---- outputs ---------------------------------------------------------------
    json_path = out_dir / f"calibrate_generator_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.hist(fwhms, bins=12, color="steelblue")
    ax.axvline(fwhm_med, color="k", ls="--", label=f"median {fwhm_med:.3f} nm")
    ax.set_xlabel("fitted FWHM (nm)"); ax.set_ylabel("lines"); ax.legend(fontsize=8); ax.set_title("(i) instrument FWHM", fontsize=9)
    ax = axes[1]
    for e in one_zone:
        ax.plot(e["l_grid"], e["S1"], marker=".", label=f"Te {e['Te']:.0f} K, Ne {e['Ne']:.0e}")
    ax.axhspan(S1_q[0], S1_q[2], color="grey", alpha=0.3, label="measured IQR")
    ax.axhline(S1_q[1], color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("l_inner (cm)"); ax.set_ylabel("S1 = resonance / weak high-Ek")
    ax.legend(fontsize=7); ax.set_title("(ii) one-zone saturation", fontsize=9)
    ax = axes[2]
    for e in two_zone:
        ax.plot(e["l_outer_grid"], e["S2"], marker=".", label=f"Te1 {e['Te1']:.0f} K, l_in {e['l_inner']:.1e}")
    ax.axhspan(S2_q[0], S2_q[2], color="grey", alpha=0.3, label="measured IQR")
    ax.axhline(S2_q[1], color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("l_outer (cm)"); ax.set_ylabel("S2 = resonance / intermediate-Ei")
    ax.legend(fontsize=7); ax.set_title("(ii) cold-shell absorption", fontsize=9)
    fig.tight_layout()
    png_path = out_dir / f"calibrate_generator_{ts}.png"
    fig.savefig(png_path, dpi=130)
    print(f"\nWrote {json_path} and {png_path}")


if __name__ == "__main__":
    main()
