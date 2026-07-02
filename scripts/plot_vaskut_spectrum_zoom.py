"""
Plot a zoomed LIBS spectrum from a VASKUT-style JSON with theoretical DB lines.

Reads intensity vs wavelength via external_data/Context/readData.py, auto-selects
the peak-densest 5 nm window (or a manual range), and draws the top 10% strongest
theoretical lines per element from LIBS_data_vacuum.db.

Usage:
    uv run python scripts/plot_vaskut_spectrum_zoom.py
    uv run python scripts/plot_vaskut_spectrum_zoom.py --wl_min 250 --wl_max 255
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.line_dictionary import compute_line_intensities_for_plasma  # noqa: E402
from external_data.Context.readData import get_spectra  # noqa: E402
from make_publication_figures import element_color_map  # noqa: E402


@dataclass
class ElementLine:
    element: str
    wavelength_nm: float
    ion_state: str
    intensity: float


def load_full_spectrum(json_path: str, run_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate CCD ranges 1+2 using readData.get_spectra (4× upsampled)."""
    w1, i1 = get_spectra(json_path, run_id=run_id, cdd_range=1)
    w2, i2 = get_spectra(json_path, run_id=run_id, cdd_range=2)
    return np.concatenate([w1, w2]), np.concatenate([i1, i2])


def find_peak_dense_window(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    window_nm: float,
    step_nm: float = 0.5,
) -> tuple[float, float]:
    """Return (wl_lo, wl_hi) for the window_nm slice with the most peaks."""
    best_count = -1
    best_lo = float(wavelength.min())
    wl_min, wl_max = float(wavelength.min()), float(wavelength.max())
    for lo in np.arange(wl_min, wl_max - window_nm, step_nm):
        hi = lo + window_nm
        mask = (wavelength >= lo) & (wavelength <= hi)
        seg = intensity[mask]
        if seg.size < 10:
            continue
        thr = float(seg.mean() + 0.5 * seg.std())
        peaks = (
            (seg[1:-1] > seg[:-2])
            & (seg[1:-1] > seg[2:])
            & (seg[1:-1] > thr)
        ).sum()
        if peaks > best_count:
            best_count = int(peaks)
            best_lo = float(lo)
    return best_lo, best_lo + window_nm


def reference_plasma(cfg_path: str) -> tuple[float, float, float, float, float]:
    """Mid-grid Te/Ne and N, C, l from line_embedding.yaml."""
    cfg = yaml.safe_load(open(cfg_path))["line_dictionary"]
    te_lo, te_hi = cfg["te_range"]
    ne_lo, ne_hi = cfg["ne_range"]
    te = 0.5 * (float(te_lo) + float(te_hi))
    ne = float(np.sqrt(float(ne_lo) * float(ne_hi)))
    return te, ne, float(cfg["N"]), float(cfg["C"]), float(cfg["l"])


def elements_in_window(db_path: str, wl_lo: float, wl_hi: float) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT Elem_name FROM QuantParam "
            "WHERE Wavelength BETWEEN ? AND ? ORDER BY Elem_name",
            (wl_lo, wl_hi),
        )
        return [r[0] for r in cur.fetchall()]


def top_lines_per_element(
    db_path: str,
    wl_lo: float,
    wl_hi: float,
    te: float,
    ne: float,
    n_density: float,
    c: float,
    optical_path: float,
    percent: float = 20.0,
    min_keep: int = 1,
) -> list[ElementLine]:
    """Top `percent`% strongest theoretical lines per element in the window."""
    lines: list[ElementLine] = []
    for elem in elements_in_window(db_path, wl_lo, wl_hi):
        try:
            wl, ion, *_rest, intensity = compute_line_intensities_for_plasma(
                elem, te, ne, n_density, c, optical_path, db_path,
            )
        except (ValueError, KeyError):
            continue
        if wl.size == 0:
            continue
        in_win = (wl >= wl_lo) & (wl <= wl_hi)
        if not in_win.any():
            continue
        win_idx = np.flatnonzero(in_win)
        intens_win = intensity[win_idx]
        n = len(win_idx)
        if n < min_keep:
            k = n
        else:
            k = max(min_keep, int(np.ceil((percent / 100.0) * n)))
        for j in np.argsort(intens_win)[::-1][:k]:
            i = int(win_idx[j])
            lines.append(ElementLine(
                element=str(elem),
                wavelength_nm=float(wl[i]),
                ion_state=str(ion[i]),
                intensity=float(intensity[i]),
            ))
    lines.sort(key=lambda x: (x.element, x.wavelength_nm))
    return lines


def plot_zoom(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    wl_lo: float,
    wl_hi: float,
    element_lines: list[ElementLine],
    run_id: int,
    auto_window: bool,
) -> plt.Figure:
    mask = (wavelength >= wl_lo) & (wavelength <= wl_hi)
    wl_zoom = wavelength[mask]
    i_zoom = intensity[mask]

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    ax.plot(wl_zoom, i_zoom, color="#4a4a4a", lw=0.8, label="Measured spectrum")

    elems = sorted({ln.element for ln in element_lines})
    colors = element_color_map(elems)

    legend_handles = []
    seen_elems: set[str] = set()
    for ln in element_lines:
        color = colors[ln.element]
        ax.axvline(ln.wavelength_nm, color=color, lw=0.9, alpha=0.75, zorder=2)
        if ln.element not in seen_elems:
            seen_elems.add(ln.element)
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2, label=ln.element),
            )

    ax.set_xlim(wl_lo, wl_hi)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    win_note = "auto peak-dense window" if auto_window else "manual window"
    ax.set_title(
        f"VASKUT spectrum zoom — run {run_id}  "
        f"({wl_lo:.1f}–{wl_hi:.1f} nm, {win_note})",
        fontsize=12,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5, 0.01,
        "Vertical lines: top theoretical transitions per element in window "
        "(10% strongest per element, LIBS_data_vacuum.db, vacuum wavelengths).",
        ha="center", fontsize=9, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json_path", default="external_data/Data/VASKUT K8.json",
    )
    parser.add_argument(
        "--db_path", default="external_data/Source/LIBS_data_vacuum.db",
    )
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--window_nm", type=float, default=5.0)
    parser.add_argument("--wl_min", type=float, default=None)
    parser.add_argument("--wl_max", type=float, default=None)
    parser.add_argument("--output_dir", default="Outputs")
    parser.add_argument(
        "--line_embedding_config", default="config/line_embedding.yaml",
    )
    parser.add_argument(
        "--line_percent", type=float, default=10.0,
        help="Keep top N%% strongest theoretical lines per element in the window",
    )
    parser.add_argument(
        "--line_min_keep", type=int, default=1,
        help="Minimum lines per element when enough transitions exist in window",
    )
    args = parser.parse_args()

    json_path = str(ROOT / args.json_path)
    db_path = str(ROOT / args.db_path)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading spectrum from {args.json_path} (run {args.run_id})...")
    wavelength, intensity = load_full_spectrum(json_path, args.run_id)

    auto_window = args.wl_min is None or args.wl_max is None
    if auto_window:
        wl_lo, wl_hi = find_peak_dense_window(wavelength, intensity, args.window_nm)
        print(f"Auto-selected window: {wl_lo:.2f}–{wl_hi:.2f} nm")
    else:
        wl_lo, wl_hi = float(args.wl_min), float(args.wl_max)
        print(f"Manual window: {wl_lo:.2f}–{wl_hi:.2f} nm")

    te, ne, n_d, c, l_path = reference_plasma(str(ROOT / args.line_embedding_config))
    print(f"Reference plasma: Te={te:.0f} K, Ne={ne:.2e} cm⁻³")

    element_lines = top_lines_per_element(
        db_path, wl_lo, wl_hi, te, ne, n_d, c, l_path,
        percent=args.line_percent, min_keep=args.line_min_keep,
    )
    n_elems = len({ln.element for ln in element_lines})
    print(
        f"Drawing {len(element_lines)} lines "
        f"(top {args.line_percent:g}% per element, {n_elems} elements)",
    )

    fig = plot_zoom(
        wavelength, intensity, wl_lo, wl_hi, element_lines,
        args.run_id, auto_window,
    )

    stem = f"vaskut_k8_spectrum_zoom_{wl_lo:.1f}_{wl_hi:.1f}nm"
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
