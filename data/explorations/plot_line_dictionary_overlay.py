"""
Overlay theoretical line centres on a synthetic spectrum (sanity check).

Usage:
  uv run python data/explorations/plot_line_dictionary_overlay.py \
    --line_embedding_config config/line_embedding.yaml \
    --libs_data_config config/libs_data_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.libs_pipeline import build_dataset_from_config
from data.line_dictionary import build_line_dictionary
from data.line_features import FEAT_VALID, build_line_features_cache
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line_embedding_config", default="config/line_embedding.yaml")
    parser.add_argument("--libs_data_config", default="config/libs_data_smoke.yaml")
    parser.add_argument("--spectrum_idx", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.line_embedding_config))
    dict_path = build_line_dictionary(cfg, project_root=ROOT, verbose=True)

    ds = build_dataset_from_config(yaml.safe_load(open(ROOT / args.libs_data_config)))
    wl = ds.wavelength
    spec = ds.spectra[args.spectrum_idx]

    feat_path = build_line_features_cache(
        ds.spectra.astype(np.float32),
        wl,
        dict_path,
        cfg["line_features"],
        spectra_cache_key=ds.cache_key,
        verbose=True,
    )

    with h5py.File(dict_path, "r") as f:
        centres = f["central_wavelength"][:]
    with h5py.File(feat_path, "r") as f:
        feats = f["features"][args.spectrum_idx]

    valid = feats[:, FEAT_VALID] > 0.5

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(wl, spec, "k-", lw=0.5, alpha=0.8)
    ax.scatter(
        centres[valid], spec[np.clip(np.searchsorted(wl, centres[valid]), 0, len(wl) - 1)],
        c="green", s=8, label=f"valid fit ({valid.sum()})",
    )
    ax.scatter(
        centres[~valid], spec[np.clip(np.searchsorted(wl, centres[~valid]), 0, len(wl) - 1)],
        c="red", s=4, alpha=0.4, label=f"failed fit ({(~valid).sum()})",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.set_title(f"Line dictionary overlay — spectrum {args.spectrum_idx}")
    out = ROOT / "data/explorations/figures/line_dictionary_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
