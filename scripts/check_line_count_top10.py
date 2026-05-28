"""
Quick diagnostic for line-dictionary selection strategies.

Compares, per element:
  - total candidate lines after Te x Ne aggregation
  - lines kept by absolute threshold (1e-33 and 1e-35 references)
  - lines kept by proposed per-element top-10% rule
    (keep all if fewer than min_keep lines)

Outputs a markdown table and aggregate totals.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.line_dictionary import (  # noqa: E402
    _list_elements,
    _te_ne_grid,
    _wavelength_clip_bounds,
    compute_line_intensities_for_plasma,
)


def _collect_best_per_element(cfg: dict, project_root: Path) -> dict[str, list[float]]:
    ld_cfg = dict(cfg["line_dictionary"])
    db_path = str((project_root / ld_cfg["db_path"]).resolve())
    te_grid, ne_grid = _te_ne_grid(ld_cfg)
    n = float(ld_cfg["N"])
    c = float(ld_cfg["C"])
    l_path = float(ld_cfg["l"])
    wmin, wmax = _wavelength_clip_bounds(ld_cfg, project_root)
    elements = _list_elements(db_path)

    out: dict[str, list[float]] = {}
    for elem in elements:
        best: dict[tuple[float, str], float] = {}
        for te in te_grid:
            for ne in ne_grid:
                wl, ion, _, _, _, _, _, intens = compute_line_intensities_for_plasma(
                    elem, float(te), float(ne), n, c, l_path, db_path
                )
                for i in range(wl.size):
                    if wmin is not None and wl[i] < wmin:
                        continue
                    if wmax is not None and wl[i] > wmax:
                        continue
                    key_line = (float(wl[i]), str(ion[i]))
                    prev = best.get(key_line)
                    cur = float(intens[i])
                    if prev is None or cur > prev:
                        best[key_line] = cur
        out[elem] = sorted(best.values(), reverse=True)
    return out


def _top_k(n: int, percent: float, min_keep: int) -> int:
    if n < min_keep:
        return n
    return max(1, int(math.ceil((percent / 100.0) * n)))


def _render_markdown(rows: list[dict[str, int]], totals: dict[str, int], cfg_path: str) -> str:
    lines = []
    lines.append("# Line-count diagnostic: threshold vs per-element top-10%")
    lines.append("")
    lines.append(f"Config: `{cfg_path}`")
    lines.append("")
    lines.append("| Element | n_total | n_thr_1e-33 | n_thr_1e-35 | n_top10pct |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['element']} | {r['n_total']} | {r['n_thr_1e33']} | {r['n_thr_1e35']} | {r['n_top10pct']} |"
        )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| total candidates | {totals['n_total']} |")
    lines.append(f"| kept by threshold 1e-33 | {totals['n_thr_1e33']} |")
    lines.append(f"| kept by threshold 1e-35 | {totals['n_thr_1e35']} |")
    lines.append(f"| kept by top-10% per element | {totals['n_top10pct']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast line-count impact check for top-10% selection.")
    parser.add_argument("--config", type=str, default="config/line_embedding.yaml")
    parser.add_argument("--out", type=str, default="external_data/cache/line_count_top10_check.md")
    parser.add_argument("--percent", type=float, default=10.0)
    parser.add_argument("--min_keep", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    best_by_elem = _collect_best_per_element(cfg, project_root)

    rows: list[dict[str, int]] = []
    totals = {"n_total": 0, "n_thr_1e33": 0, "n_thr_1e35": 0, "n_top10pct": 0}
    for elem in sorted(best_by_elem.keys()):
        vals = best_by_elem[elem]
        n_total = len(vals)
        n_33 = sum(1 for v in vals if v >= 1.0e-33)
        n_35 = sum(1 for v in vals if v >= 1.0e-35)
        n_top = _top_k(n_total, args.percent, args.min_keep)
        rows.append(
            {
                "element": elem,
                "n_total": n_total,
                "n_thr_1e33": n_33,
                "n_thr_1e35": n_35,
                "n_top10pct": n_top,
            }
        )
        totals["n_total"] += n_total
        totals["n_thr_1e33"] += n_33
        totals["n_thr_1e35"] += n_35
        totals["n_top10pct"] += n_top

    md = _render_markdown(rows, totals, args.config)
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    print(md)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
