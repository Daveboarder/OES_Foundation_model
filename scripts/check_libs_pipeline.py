"""
Sanity checks for the physics-based LIBS data pipeline.

Generates a small batch via `data.libs_pipeline.build_dataset_from_config`,
then:
  1. Plots every spectrum (one panel each) with its sample name + (Te, Ne) in
     the title, plus an overlay panel.
  2. Verifies Te and Ne samples fall inside the configured ranges (assertion).
  3. Verifies per-row concentrations sum to ~1 after normalisation (assertion).
  4. Reports basic spectrum stats (nonzero bins, dynamic range, peak count).

Outputs land in `sanity_checks/<timestamp>/`.

Usage:
    uv run python scripts/check_libs_pipeline.py
    uv run python scripts/check_libs_pipeline.py --libs_data_config config/libs_data_smoke.yaml
    uv run python scripts/check_libs_pipeline.py --n_types 10 --n_per_type 3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.libs_pipeline import build_dataset_from_config, load_wavelength


def _format_ne(x: float) -> str:
    return f"{x:.2e}"


def plot_spectra(spectra: np.ndarray, wavelength: np.ndarray, names: list[str],
                 te: np.ndarray, ne: np.ndarray, out_path: Path) -> None:
    n = len(spectra)
    fig, axes = plt.subplots(n + 1, 1, figsize=(13, 2.3 * (n + 1)), sharex=True)
    if n == 0:
        return

    # One panel per spectrum
    for i in range(n):
        ax = axes[i]
        ax.plot(wavelength, spectra[i], linewidth=0.6, color="steelblue")
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("I (norm)")
        ax.set_title(
            f"{names[i]}   Te={te[i]:.0f} K   Ne={_format_ne(ne[i])} cm⁻³",
            fontsize=9, loc="left",
        )
        ax.grid(alpha=0.2, linewidth=0.5)

    # Overlay panel
    ax = axes[-1]
    for i in range(n):
        ax.plot(wavelength, spectra[i], linewidth=0.5, alpha=0.7, label=names[i])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("I (norm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title("Overlay", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def check_ranges(table, te_range, ne_range) -> tuple[bool, list[str]]:
    """Verify Te/Ne fall inside the configured ranges; return (ok, messages)."""
    msgs: list[str] = []
    te = table["Te"].values
    ne = table["Ne"].values

    te_lo, te_hi = te_range
    ne_lo, ne_hi = ne_range

    te_ok = np.all((te >= te_lo) & (te <= te_hi))
    ne_ok = np.all((ne >= ne_lo) & (ne <= ne_hi))

    msgs.append(f"Te range configured: [{te_lo}, {te_hi}] K")
    msgs.append(f"Te samples:          min={te.min():.1f}, max={te.max():.1f}, mean={te.mean():.1f}")
    msgs.append(f"   in-range: {te_ok}   ({(te>=te_lo).sum()}/{len(te)} >= lo, {(te<=te_hi).sum()}/{len(te)} <= hi)")

    msgs.append(f"Ne range configured: [{_format_ne(ne_lo)}, {_format_ne(ne_hi)}] cm^-3")
    msgs.append(f"Ne samples:          min={_format_ne(ne.min())}, max={_format_ne(ne.max())}, mean={_format_ne(ne.mean())}")
    msgs.append(f"   in-range: {ne_ok}   ({(ne>=ne_lo).sum()}/{len(ne)} >= lo, {(ne<=ne_hi).sum()}/{len(ne)} <= hi)")

    # Coverage warning: tiny sample sizes won't span the full range; that's fine,
    # but worth surfacing so the user doesn't misread "didn't reach max" as a bug.
    te_span = (te.max() - te.min()) / (te_hi - te_lo)
    ne_span = np.log10(ne.max() / ne.min()) / np.log10(ne_hi / ne_lo) if ne.min() > 0 else 0
    msgs.append(f"Coverage: Te spans {te_span:.0%} of configured range, "
                f"Ne (log) spans {ne_span:.0%}.")

    return te_ok and ne_ok, msgs


def check_concentrations(table) -> tuple[bool, list[str]]:
    """Per-sample element concentrations should sum to ~1.0 after normalisation."""
    skip = {"Te", "Ne", "sample_type_id", "sample_type_name", "unique_id"}
    elem_cols = [c for c in table.columns if c not in skip]
    sums = table[elem_cols].sum(axis=1).values
    ok = np.allclose(sums, 1.0, atol=1e-6)
    msgs = [
        f"Per-sample concentration sums: min={sums.min():.6f}, max={sums.max():.6f}, "
        f"mean={sums.mean():.6f} (target 1.0)",
        f"   all-equal-to-1: {ok}",
    ]
    return ok, msgs


def check_spectrum_stats(spectra: np.ndarray) -> list[str]:
    nonzero = (spectra > 1e-6).sum(axis=1)
    return [
        f"Spectrum stats over {spectra.shape[0]} shots × {spectra.shape[1]} bins:",
        f"   nonzero bins per spectrum: min={nonzero.min()}, max={nonzero.max()}, mean={nonzero.mean():.0f}",
        f"   intensity range:           [{spectra.min():.3f}, {spectra.max():.3f}]   (expected [0, 1])",
        f"   mean intensity per shot:   min={spectra.mean(axis=1).min():.4f}, max={spectra.mean(axis=1).max():.4f}",
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs_data_config", default="config/libs_data_smoke.yaml",
                   help="Path to LIBS data pipeline config")
    p.add_argument("--n_types", type=int, default=None,
                   help="Override max_sample_types (default: from config)")
    p.add_argument("--n_per_type", type=int, default=None,
                   help="Override n_samples_per_type (default: from config)")
    p.add_argument("--out_dir", default="sanity_checks",
                   help="Base output directory")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.libs_data_config))
    cfg.setdefault("generation", {})
    if args.n_types is not None:
        cfg["generation"]["max_sample_types"] = args.n_types
    if args.n_per_type is not None:
        cfg["generation"]["n_samples_per_type"] = args.n_per_type

    te_range = tuple(cfg["ranges"]["te"])
    ne_range = tuple(cfg["ranges"]["ne"])

    print("=" * 70)
    print("LIBS pipeline sanity checks")
    print("=" * 70)
    print(f"Config: {args.libs_data_config}")
    print(f"  Te range: {te_range}")
    print(f"  Ne range: {ne_range}")
    print(f"  max_sample_types:    {cfg['generation'].get('max_sample_types')}")
    print(f"  n_samples_per_type:  {cfg['generation'].get('n_samples_per_type')}")
    print()

    ds = build_dataset_from_config(cfg)
    if len(ds) == 0:
        print("ERROR: dataset is empty. Check sample matrix vs DB element coverage.")
        sys.exit(1)

    wavelength = ds.wavelength
    table = ds.sample_table
    spectra = ds.spectra
    names = table["sample_type_name"].astype(str).tolist()

    out_root = Path(args.out_dir) / datetime.now().strftime("libs_pipeline_%Y-%m-%d_%H-%M-%S")
    out_root.mkdir(parents=True, exist_ok=True)

    # --- 1. Plot --------------------------------------------------------------
    plot_path = out_root / "spectra.png"
    plot_spectra(spectra, wavelength, names, table["Te"].values, table["Ne"].values, plot_path)
    print(f"[plot]    saved {plot_path}   ({len(spectra)} spectra, {len(wavelength)} bins)")

    # --- 2-4. Checks ---------------------------------------------------------
    report_lines = ["LIBS pipeline sanity report", "=" * 40, ""]
    report_lines.append(f"Wavelength array: {len(wavelength)} bins, "
                        f"[{wavelength.min():.2f}, {wavelength.max():.2f}] nm")
    report_lines.append("")

    ok_ranges, msgs = check_ranges(table, te_range, ne_range)
    report_lines.extend(msgs); report_lines.append("")
    ok_concs, msgs = check_concentrations(table)
    report_lines.extend(msgs); report_lines.append("")
    report_lines.extend(check_spectrum_stats(spectra)); report_lines.append("")

    overall = ok_ranges and ok_concs
    report_lines.append(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    report_path = out_root / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[report]  saved {report_path}")
    print()
    print("\n".join(report_lines))

    sys.exit(0 if overall else 2)


if __name__ == "__main__":
    main()
