"""
Publication-quality figures for the LIBS foundation model.

Reads the data caches + outputs of a fine-tuned `quantification_binned` run
(line_token_linear embedding) and renders a PowerPoint-ready figure set:
300 dpi PNG + editable-text SVG, white background, large fonts.

Figures
    fig1_annotated_spectrum      representative spectrum, top attention lines labeled
    fig2_importance_vs_spectrum  mean spectrum + per-line CLS attention (shared x)
    fig3a_element_attention      element-to-element self-attention heatmap
    fig3b_line_pair_attention    line-line self-attention among top lines
    fig4_pred_vs_true            decoded concentration scatter grid (inference)
    fig4b_per_element_r2         per-element test R^2 bar chart (from run_info)
    fig5_training_curves         pretrain + finetune curves from TensorBoard logs
    fig6_embedding_map           t-SNE of pooled embeddings colored by Fe content
    fig7_graphical_abstract      composite 16:9 panel (a-d)

Animations (GIF)
    anim_attention_layers.gif    CLS attention per transformer layer
    anim_line_buildup.gif        top lines appearing one by one on the spectrum

Usage:
    uv run python make_publication_figures.py \
        --run_dir runs/finetune_2026-06-04_21-03-15_libs_binned_ft

    # quick re-render of selected figures only
    uv run python make_publication_figures.py --only fig1,fig3,gif_buildup

    # no checkpoint inference (skips fig4, fig6, gif_layers, abstract panels c/d)
    uv run python make_publication_figures.py --skip-inference
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from analyze_attention_importance import (  # noqa: E402
    _checkpoint_encoder_state,
    _drop_zero_valid_spectra,
    _normalize_line_attention,
    build_encoder,
    cls_attention_per_layer,
)
from data.libs_pipeline import load_wavelength  # noqa: E402
from data.line_features import fwhm_voigt, voigt  # noqa: E402
from data.line_tokenization import FEATURE_NAMES  # noqa: E402
from models.heads import bin_to_concentration, concentration_to_presence  # noqa: E402
from training.finetune import LIBSFinetuneModule  # noqa: E402
from utils.run_manager import RunManager  # noqa: E402

DEFAULT_RUN = "runs/finetune_2026-06-04_21-03-15_libs_binned_ft"

ALL_TARGETS = [
    "fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
    "gif_layers", "gif_buildup",
]

# ────────────────────────────────────────────────────────────────────────────
# Publication style
# ────────────────────────────────────────────────────────────────────────────

PUB_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "axes.linewidth": 1.1,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Keep text as text in SVG so it stays editable in PowerPoint/Illustrator.
    "svg.fonttype": "none",
}

SPECTRUM_COLOR = "#4a4a4a"
ACCENT = "#c1272d"

# Consistent element colors across every figure (Okabe-Ito + tab20 fallback).
_BASE_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#7f3c8d", "#11A579", "#E73F74",
    "#3969AC", "#80BA5A", "#E68310", "#008695", "#CF1C90",
]


class FigureRegistry:
    """Saves figures as PNG + SVG and records captions for FIGURES_README.txt."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.entries: list[tuple[str, str]] = []

    def save(self, fig, name: str, caption: str, svg: bool = True):
        png = self.output_dir / f"{name}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        if svg:
            fig.savefig(self.output_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        self.entries.append((f"{name}.png" + (" / .svg" if svg else ""), caption))
        print(f"Saved: {png.name}" + (" (+ svg)" if svg else ""))

    def add_file(self, filename: str, caption: str):
        self.entries.append((filename, caption))

    def _existing_entries(self, path: Path) -> dict[str, str]:
        """Parse a previous README so partial re-runs keep the full index."""
        if not path.is_file():
            return {}
        entries: dict[str, str] = {}
        current = None
        for raw in path.read_text().splitlines():
            if raw.startswith("    ") and current:
                entries[current] = (entries[current] + " " + raw.strip()).strip()
            elif raw and not raw.startswith((" ", "=")) and (
                    ".png" in raw or ".gif" in raw):
                current = raw.strip()
                entries[current] = ""
            elif not raw:
                current = None
        return entries

    def write_readme(self, header_lines: list[str]):
        path = self.output_dir / "FIGURES_README.txt"
        merged = self._existing_entries(path)
        for name, caption in self.entries:
            merged[name] = caption
        # Keep only entries whose file still exists in the output folder.
        merged = {
            name: cap for name, cap in merged.items()
            if (self.output_dir / name.split(" /")[0].strip()).is_file()
        }
        with open(path, "w") as f:
            f.write("Publication figures\n" + "=" * 60 + "\n")
            for line in header_lines:
                f.write(line + "\n")
            f.write("\n")
            for name in sorted(merged):
                f.write(f"{name}\n    {merged[name]}\n\n")
        print(f"Saved: {path.name}")


def element_color_map(elements: list[str]) -> dict[str, str]:
    uniq = list(dict.fromkeys(elements))
    return {e: _BASE_COLORS[i % len(_BASE_COLORS)] for i, e in enumerate(uniq)}


# ────────────────────────────────────────────────────────────────────────────
# Asset loading (lazy; everything reads caches/CSVs directly)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_cache_path(recorded: str | None, cache_dir: Path, pattern: str) -> Path:
    """Resolve a possibly machine-specific recorded path against the local cache."""
    if recorded:
        p = Path(recorded)
        if p.is_file():
            return p
        local = cache_dir / p.name
        if local.is_file():
            return local
    candidates = sorted(cache_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no cache file matching {pattern} in {cache_dir}")
    return candidates[0]


class Assets:
    """Lazy loader for run artifacts, caches and model inference results."""

    def __init__(self, run_dir: Path, args):
        self.args = args
        self.run_dir = run_dir
        self.config = yaml.safe_load(open(run_dir / "config.yaml"))
        self.run_info = yaml.safe_load(open(run_dir / "run_info.yaml"))
        self.element_names: list[str] = list(self.run_info["element_names"])
        self.task: str = str(self.run_info.get("task", "quantification_binned"))
        self.cache_dir = Path("external_data/cache")
        self._lod_vector: np.ndarray | None = None

        self._token_meta = None
        self._wavelength = None
        self._spectra_file = None
        self._splits = None
        self._concentrations = None
        self._encoder = None
        self._module = None
        self._inference = None
        self._spectrum_sample = None

        # Newest attention-importance evaluation folder.
        att_dirs = sorted((run_dir / "evaluation").glob("attention_importance_*"))
        if not att_dirs:
            raise FileNotFoundError(
                f"no attention_importance_* folder under {run_dir}/evaluation — "
                "run analyze_attention_importance.py first"
            )
        self.attention_dir = att_dirs[-1]
        print(f"Attention CSVs: {self.attention_dir}")

        self.per_line = pd.read_csv(self.attention_dir / "per_line_importance.csv")
        self.per_element = pd.read_csv(self.attention_dir / "per_element_importance.csv")
        self.pair_csv = self.attention_dir / "line_pair_attention.csv"
        self.elem_matrix_csv = self.attention_dir / "element_attention_matrix.csv"

        self.tokens_path = _resolve_cache_path(
            self.run_info.get("line_tokens_path"), self.cache_dir, "line_tokens_*.h5",
        )
        libs_cfg = yaml.safe_load(open(self.run_info["libs_data_config"]))
        self.wavelength_json = libs_cfg["paths"]["wavelength_json"]

        # Voigt-fit parameters used by the token pipeline (for fig8).
        self.fit_cfg = {"window_nm": 0.3, "gamma_init": 0.1, "sigma_init": 0.006}
        lec = self.run_info.get("line_embedding_config")
        if lec and Path(lec).is_file():
            le = yaml.safe_load(open(lec)).get("line_features", {})
            for k in self.fit_cfg:
                if k in le:
                    self.fit_cfg[k] = float(le[k])

    # ── tokens / wavelength / spectra ──
    @property
    def token_meta(self) -> dict:
        if self._token_meta is None:
            with h5py.File(self.tokens_path, "r") as f:
                self._token_meta = {
                    "n_lines": int(f.attrs["n_lines"]),
                    "n_features": int(f.attrs["n_features"]),
                    "n_spectra": int(f.attrs["n_spectra"]),
                    "feature_names": FEATURE_NAMES,
                    "feature_mean": np.asarray(f.attrs["feature_mean"], dtype=np.float32),
                    "feature_std": np.asarray(f.attrs["feature_std"], dtype=np.float32),
                    "central_wavelength": f["central_wavelength"][:].astype(np.float32),
                }
        return self._token_meta

    @property
    def wavelength(self) -> np.ndarray:
        if self._wavelength is None:
            self._wavelength = load_wavelength(self.wavelength_json)
        return self._wavelength

    @property
    def spectra_h5(self) -> h5py.File:
        if self._spectra_file is None:
            path = _resolve_cache_path(None, self.cache_dir, "synthetic_cache_*.h5")
            # Prefer the cache whose splits file matches the trained sample counts.
            for cand in sorted(self.cache_dir.glob("synthetic_cache_*.h5")):
                with h5py.File(cand, "r") as f:
                    if f["spectra"].shape[0] == self.n_total_spectra:
                        path = cand
                        break
            self._spectra_file = h5py.File(path, "r")
            print(f"Spectra cache: {path.name} {self._spectra_file['spectra'].shape}")
        return self._spectra_file

    @property
    def n_total_spectra(self) -> int:
        return (self.run_info["train_samples"] + self.run_info["val_samples"]
                + self.run_info["test_samples"])

    @property
    def splits(self) -> dict[str, np.ndarray]:
        if self._splits is None:
            cands = sorted(self.cache_dir.glob("splits_*.json"))
            chosen = None
            for cand in cands:
                s = json.load(open(cand))
                if (len(s.get("test", [])) == self.run_info["test_samples"]
                        and sum(len(v) for v in s.values()) == self.n_total_spectra):
                    chosen = s
                    break
            if chosen is None:
                raise FileNotFoundError("no splits_*.json matching the run sample counts")
            self._splits = {k: np.asarray(v, dtype=np.int64) for k, v in chosen.items()}
        return self._splits

    @property
    def concentrations(self) -> np.ndarray:
        """[n_spectra, n_elements] in run_info element order (matches training)."""
        if self._concentrations is None:
            g = self.spectra_h5["sample_table"]
            cols = [np.asarray(g[name], dtype=np.float32) for name in self.element_names]
            conc = np.stack(cols, axis=1)
            np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            self._concentrations = np.clip(conc, 0.0, 1.0)
        return self._concentrations

    # ── representative + mean spectrum ──
    def spectrum_sample(self, n_sub: int = 512):
        """(representative_spectrum, mean_spectrum) over a test-split subsample."""
        if self._spectrum_sample is None:
            rng = np.random.default_rng(self.args.seed)
            test_idx = self.splits["test"]
            sub = np.sort(rng.choice(test_idx, size=min(n_sub, len(test_idx)),
                                     replace=False))
            block = self.spectra_h5["spectra"][sub, :].astype(np.float32)
            mean_spec = block.mean(axis=0)
            totals = block.sum(axis=1)
            rep_pos = int(np.argsort(totals)[len(totals) // 2])
            self._spectrum_sample = (block[rep_pos], mean_spec, int(sub[rep_pos]))
        return self._spectrum_sample

    # ── model ──
    @property
    def device(self) -> str:
        if self.args.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.args.device

    def finetune_checkpoint(self) -> Path:
        """Prefer best.ckpt (best validation metric) over last.ckpt."""
        best = self.run_dir / "checkpoints" / "best.ckpt"
        if best.is_file():
            return best
        ckpt = RunManager.from_existing_run(str(self.run_dir)) \
            .get_checkpoint_for_mode("finetune")
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint in {self.run_dir}")
        return Path(ckpt)

    @property
    def encoder(self):
        if self._encoder is None:
            cfg = self.config
            cfg["data"]["n_bins"] = self.token_meta["n_lines"]
            cfg["model"]["max_seq_len"] = self.token_meta["n_lines"] + 1
            ckpt = self.finetune_checkpoint()
            print(f"Loading encoder from {ckpt}")
            enc = build_encoder(cfg, self.run_info, self.token_meta)
            state = _checkpoint_encoder_state(str(ckpt))
            model_sd = enc.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_sd and model_sd[k].shape == v.shape}
            enc.load_state_dict(filtered, strict=False)
            self._encoder = enc.to(self.device).eval()
        return self._encoder

    @property
    def module(self) -> LIBSFinetuneModule:
        """Full finetune module (encoder + binned head) loaded from best.ckpt."""
        if self._module is None:
            module = LIBSFinetuneModule(
                encoder=self.encoder,
                task=self.run_info["task"],
                n_classes=self.config["data"]["n_classes"],
                n_elements=self.run_info["n_elements"],
                n_concentration_bins=self.run_info["n_concentration_bins"],
                pool=self.run_info["pool"],
                element_names=self.element_names,
            )
            ckpt_path = self.finetune_checkpoint()
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model_sd = module.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in model_sd and model_sd[k].shape == v.shape}
            missing = [k for k in model_sd if k not in filtered]
            module.load_state_dict(filtered, strict=False)
            n_head = sum(
                1 for k in filtered
                if k.startswith(("binned_head.", "detection_head."))
            )
            print(f"Loaded full module: {len(filtered)} tensors "
                  f"({n_head} head tensors, {len(missing)} missing)")
            self._module = module.to(self.device).eval()
        return self._module

    @property
    def lod_vector(self) -> torch.Tensor:
        """Per-element LOD mass fractions aligned with element_names."""
        if self._lod_vector is None:
            lod_map = self.run_info.get("element_lod") or {}
            default = float(self.run_info.get("default_lod", 1e-4))
            self._lod_vector = np.array(
                [float(lod_map.get(name, default)) for name in self.element_names],
                dtype=np.float32,
            )
        return torch.from_numpy(self._lod_vector)

    # ── inference on the test split ──
    @torch.no_grad()
    def inference(self) -> dict:
        """Decoded predictions + pooled embeddings on a test-split subsample."""
        if self._inference is not None:
            return self._inference
        rng = np.random.default_rng(self.args.seed)
        test_idx = self.splits["test"]
        n_take = min(self.args.max_samples, len(test_idx))
        sub = np.sort(rng.choice(test_idx, size=n_take, replace=False))
        targets_all = self.concentrations[sub]

        module = self.module
        device = self.device
        n_bins = self.run_info["n_concentration_bins"]
        lod = self.lod_vector.to(device)

        preds, probs, reprs, kept_targets, kept_conc = [], [], [], [], []
        bs = self.args.batch_size
        print(f"Running inference on {n_take} test spectra "
              f"(batch_size={bs}, device={device}, task={self.task})...")
        with h5py.File(self.tokens_path, "r") as f:
            tok_ds, valid_ds = f["tokens"], f["fit_valid"]
            for start in range(0, n_take, bs):
                idx = sub[start:start + bs]
                tokens = torch.from_numpy(tok_ds[idx].astype(np.float32))
                valid = torch.from_numpy(valid_ds[idx].astype(np.uint8))
                keep = valid.sum(dim=1) > 0
                if not keep.any():
                    continue
                batch = {
                    "tokens": tokens[keep].to(device),
                    "fit_valid": valid[keep].to(device),
                }
                out = module(batch)
                conc_batch = torch.from_numpy(
                    targets_all[start:start + bs][keep.numpy()].astype(np.float32),
                ).to(device)
                if self.task == "detection":
                    preds.append(out["presence_pred"].cpu().numpy())
                    probs.append(out["presence_prob"].cpu().numpy())
                    kept_targets.append(
                        concentration_to_presence(conc_batch, lod).cpu().numpy(),
                    )
                else:
                    bin_pred = out["bin_logits"].argmax(dim=-1)
                    preds.append(
                        bin_to_concentration(bin_pred, n_bins=n_bins).cpu().numpy(),
                    )
                    kept_targets.append(conc_batch.cpu().numpy())
                reprs.append(out["representation"].float().cpu().numpy())
                kept_conc.append(conc_batch.cpu().numpy())
                done = min(start + bs, n_take)
                if (start // bs) % 10 == 9:
                    print(f"  {done}/{n_take}")

        self._inference = {
            "preds": np.concatenate(preds, axis=0),
            "targets": np.concatenate(kept_targets, axis=0),
            "concentrations": np.concatenate(kept_conc, axis=0),
            "representations": np.concatenate(reprs, axis=0),
        }
        if self.task == "detection":
            self._inference["probs"] = np.concatenate(probs, axis=0)
        print(f"  done: {self._inference['preds'].shape[0]} spectra kept")
        return self._inference

    def close(self):
        if self._spectra_file is not None:
            self._spectra_file.close()


# ────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers (reused by standalone figures and the abstract)
# ────────────────────────────────────────────────────────────────────────────

def top_lines_table(assets: Assets, n: int) -> pd.DataFrame:
    df = assets.per_line.sort_values("importance_layer_mean", ascending=False).head(n)
    return df.reset_index(drop=True)


def spread_positions(xs: np.ndarray, lo: float, hi: float,
                     min_dx: float, n_iter: int = 400) -> np.ndarray:
    """1D label fan-out: keep positions near xs but at least min_dx apart."""
    order = np.argsort(xs)
    pos = xs[order].astype(np.float64).copy()
    for _ in range(n_iter):
        moved = False
        for i in range(1, len(pos)):
            gap = pos[i] - pos[i - 1]
            if gap < min_dx:
                shift = (min_dx - gap) / 2
                pos[i - 1] -= shift
                pos[i] += shift
                moved = True
        np.clip(pos, lo, hi, out=pos)
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def annotate_lines(ax, lines: pd.DataFrame, wl: np.ndarray, spec: np.ndarray,
                   elem_colors: dict, ymax: float, k_visible: int | None = None,
                   fontsize: float = 9.5, marker_scale: float = 1.0):
    """Stems + fanned-out rotated labels with leader lines for the top lines.

    Label x-positions are computed for the FULL `lines` table so animations
    that reveal lines incrementally (k_visible) keep labels in place.
    """
    n = len(lines)
    wls = lines["central_wavelength_nm"].to_numpy(dtype=np.float64)
    imp = lines["importance_layer_mean"].to_numpy()
    imp_norm = imp / max(imp.max(), 1e-12)

    x0, x1 = wl.min() - 5, wl.max() + 5
    min_dx = (x1 - x0) * 0.016
    label_x = spread_positions(wls, x0 + min_dx, x1 - min_dx, min_dx)
    y_leader = ymax * 1.13   # where leader lines end and label text begins

    k = n if k_visible is None else min(k_visible, n)
    for i in range(k):
        row = lines.iloc[i]
        lwl = float(wls[i])
        elem = str(row["element"])
        color = elem_colors.get(elem, ACCENT)
        j = int(np.argmin(np.abs(wl - lwl)))
        peak = float(spec[max(0, j - 3): j + 4].max())
        tip = peak + 0.045 * ymax
        ax.plot([lwl, lwl], [peak + 0.012 * ymax, tip], color=color,
                lw=1.5, alpha=0.95, zorder=3)
        ax.plot(lwl, tip, marker="v",
                ms=(3.5 + 4.5 * imp_norm[i]) * marker_scale, color=color, zorder=4)
        # Leader from stem tip to the fanned-out label position.
        ax.plot([lwl, label_x[i]], [tip + 0.01 * ymax, y_leader],
                color=color, lw=0.7, alpha=0.6, zorder=3, clip_on=False)
        ax.text(label_x[i], y_leader + 0.015 * ymax, f"{elem} {lwl:.1f}",
                fontsize=fontsize, color=color, ha="center", va="bottom",
                rotation=90, rotation_mode="anchor", clip_on=False, zorder=5)

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, ymax * 1.55)


def draw_annotated_spectrum(ax, assets: Assets, top_n: int, elem_colors: dict,
                            label_fontsize: float = 9.5,
                            lines_subset: pd.DataFrame | None = None,
                            marker_scale: float = 1.0):
    """Spectrum + labeled vertical markers at the top attention lines."""
    wl = assets.wavelength
    spec, _, _ = assets.spectrum_sample()
    lines = lines_subset if lines_subset is not None else top_lines_table(assets, top_n)

    ax.plot(wl, spec, lw=0.7, color=SPECTRUM_COLOR, zorder=2)
    ymax = float(spec.max())
    annotate_lines(ax, lines, wl, spec, elem_colors, ymax,
                   fontsize=label_fontsize, marker_scale=marker_scale)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    return lines


def draw_element_heatmap(ax, assets: Assets, fig, cbar: bool = True,
                         tick_fontsize: float = 8.5,
                         top_elements: int | None = None):
    df = pd.read_csv(assets.elem_matrix_csv)
    symbols = list(df.columns[1:])
    M = df[symbols].to_numpy(dtype=np.float64)
    # Drop elements whose query AND key attention is essentially zero
    # (no valid lines) — they would render as black bands on a log scale.
    floor = M.max() * 1e-6
    keep = (M.max(axis=1) > floor) | (M.max(axis=0) > floor)
    M = M[np.ix_(keep, keep)]
    symbols = [s for s, k in zip(symbols, keep) if k]
    if top_elements is not None and top_elements < len(symbols):
        # Subset to the most CLS-attended elements (keeps small panels legible).
        ranked = list(assets.per_element.sort_values(
            "summed_importance", ascending=False)["element"])
        chosen = [s for s in ranked if s in symbols][:top_elements]
        idx = [symbols.index(s) for s in chosen]
        M = M[np.ix_(idx, idx)]
        symbols = chosen
    vals = np.maximum(M, 1e-12)
    pos = vals[vals > floor]
    vmin = float(np.percentile(pos, 1)) if pos.size else 1e-8
    im = ax.imshow(vals, cmap="magma", aspect="equal",
                   norm=LogNorm(vmin=vmin, vmax=vals.max()))
    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=tick_fontsize)
    ax.set_xlabel("Key element (attended to)")
    ax.set_ylabel("Query element (attending)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    if cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label="Mean attention (log scale)")
    return im


def draw_pred_scatter(ax, y_true: np.ndarray, y_pred: np.ndarray, elem: str,
                      color: str = "#0072B2", show_xlabel: bool = True,
                      show_ylabel: bool = True, percent: bool = True):
    scale = 100.0 if percent else 1.0
    t, p = y_true * scale, y_pred * scale
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", lw=1.0,
            ls="--", zorder=1)
    # rasterized: keeps SVG small (axes/text stay vector, points become raster).
    ax.scatter(t, p, s=7, alpha=0.25, color=color, edgecolors="none", zorder=2,
               rasterized=True)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")

    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.mean(np.abs(t - p)))
    unit = "wt.%" if percent else ""
    ax.text(0.04, 0.96, f"{elem}\n$R^2$ = {r2:.3f}\nMAE = {mae:.3g} {unit}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5)
    if show_xlabel:
        ax.set_xlabel(f"True ({unit})" if unit else "True")
    if show_ylabel:
        ax.set_ylabel(f"Predicted ({unit})" if unit else "Predicted")
    return r2


def draw_embedding_map(ax, fig, reprs: np.ndarray, color_values: np.ndarray,
                       seed: int, cbar: bool = True, max_points: int = 3000,
                       color_label: str = "Fe content (wt.%)"):
    from sklearn.manifold import TSNE
    n = min(max_points, reprs.shape[0])
    rng = np.random.default_rng(seed)
    pick = rng.choice(reprs.shape[0], size=n, replace=False)
    X = reprs[pick].astype(np.float64)
    emb = TSNE(n_components=2, random_state=seed, perplexity=min(30, n // 4),
               init="pca").fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_values[pick] * 100, s=8,
                    cmap="viridis", alpha=0.8, edgecolors="none",
                    rasterized=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label=color_label)
    return sc


def fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[..., :3].copy())


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────

def make_fig1(assets: Assets, reg: FigureRegistry, top_n: int):
    lines = top_lines_table(assets, top_n)
    elem_colors = element_color_map(list(lines["element"]))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    draw_annotated_spectrum(ax, assets, top_n, elem_colors)

    # Inset zoom on the window holding the most top lines.
    wls = lines["central_wavelength_nm"].to_numpy()
    width = 14.0
    best_lo, best_count = wls.min(), 0
    for lo in wls:
        count = int(((wls >= lo) & (wls <= lo + width)).sum())
        if count > best_count:
            best_lo, best_count = lo, count
    if best_count >= 3:
        wl = assets.wavelength
        spec, _, _ = assets.spectrum_sample()
        lo, hi = best_lo - 1.5, best_lo + width + 1.5
        axins = ax.inset_axes([0.05, 0.30, 0.27, 0.38])
        m = (wl >= lo) & (wl <= hi)
        axins.plot(wl[m], spec[m], lw=0.9, color=SPECTRUM_COLOR)
        for _, row in lines.iterrows():
            lwl = float(row["central_wavelength_nm"])
            if lo <= lwl <= hi:
                axins.axvline(lwl, color=elem_colors.get(str(row["element"]), ACCENT),
                              lw=1.2, alpha=0.85)
        axins.set_xlim(lo, hi)
        axins.set_yticks([])
        axins.tick_params(labelsize=8.5)
        for side in ("top", "right"):
            axins.spines[side].set_visible(True)
        ax.indicate_inset_zoom(axins, edgecolor="0.45", lw=1.0)

    fig.tight_layout()
    reg.save(fig, "fig1_annotated_spectrum",
             f"Representative test spectrum with the {top_n} most important "
             "emission lines (by mean CLS attention of the fine-tuned model) "
             "marked and labeled as element + wavelength. Marker size encodes "
             "attention importance; inset zooms on the densest line region.")


def make_fig2(assets: Assets, reg: FigureRegistry, top_n: int):
    wl_lines = assets.per_line.sort_values("line_index")
    line_wl = wl_lines["central_wavelength_nm"].to_numpy()
    line_imp = wl_lines["importance_layer_mean"].to_numpy()
    lines = top_lines_table(assets, top_n)
    elem_colors = element_color_map(list(lines["element"]))
    _, mean_spec, _ = assets.spectrum_sample()
    wl = assets.wavelength

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.0, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.5], "hspace": 0.10},
    )
    ax1.plot(wl, mean_spec, lw=0.7, color=SPECTRUM_COLOR)
    ax1.set_ylabel("Mean intensity (a.u.)")
    ax1.set_title("Mean test spectrum and per-line attention importance "
                  "of the fine-tuned model")

    ax2.vlines(line_wl, 0, line_imp, color="0.62", lw=0.8)
    imax = float(line_imp.max())
    x0, x1 = wl.min() - 5, wl.max() + 5
    wls = lines["central_wavelength_nm"].to_numpy(dtype=np.float64)
    min_dx = (x1 - x0) * 0.016
    label_x = spread_positions(wls, x0 + min_dx, x1 - min_dx, min_dx)
    y_leader = imax * 1.16
    for i, row in lines.iterrows():
        lwl = float(row["central_wavelength_nm"])
        imp = float(row["importance_layer_mean"])
        elem = str(row["element"])
        color = elem_colors.get(elem, ACCENT)
        ax2.vlines(lwl, 0, imp, color=color, lw=1.8)
        ax2.plot(lwl, imp, "o", ms=5, color=color)
        ax2.plot([lwl, label_x[i]], [imp + 0.02 * imax, y_leader],
                 color=color, lw=0.7, alpha=0.6, clip_on=False)
        ax2.text(label_x[i], y_leader + 0.02 * imax, f"{elem} {lwl:.1f}",
                 fontsize=9.5, color=color, ha="center", va="bottom",
                 rotation=90, rotation_mode="anchor", clip_on=False)
    ax2.set_ylim(0, imax * 1.65)
    ax2.set_xlim(x0, x1)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("CLS attention importance")
    fig.align_ylabels([ax1, ax2])
    reg.save(fig, "fig2_importance_vs_spectrum",
             "Mean test spectrum (top) aligned with the per-line CLS attention "
             f"importance of all {len(line_wl)} line tokens (bottom); the "
             f"{top_n} most attended lines are highlighted and labeled.")


def make_fig3(assets: Assets, reg: FigureRegistry):
    # 3a — element-to-element attention.
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    draw_element_heatmap(ax, assets, fig)
    ax.set_title("Element-to-element self-attention\n"
                 "(mean attention a line of the query element pays to lines "
                 "of the key element)", fontsize=12.5)
    reg.save(fig, "fig3a_element_attention",
             "Element-to-element token self-attention matrix (head- and "
             "layer-averaged, log color scale). Diagonal = cross-referencing "
             "lines of the same species; off-diagonal = learned co-occurrence "
             "or spectral interference.")

    # 3b — line-pair attention among top lines.
    if not assets.pair_csv.is_file():
        print("line_pair_attention.csv not found — skipping fig3b")
        return
    pairs = pd.read_csv(assets.pair_csv)
    order = (assets.per_line.sort_values("importance_layer_mean", ascending=False)
             ["line_index"].tolist())
    in_pairs = set(pairs["query_line_index"]) | set(pairs["key_line_index"])
    top_idx = [i for i in order if i in in_pairs][:25]
    pos = {li: k for k, li in enumerate(top_idx)}
    K = len(top_idx)
    M = np.zeros((K, K))
    labels = [None] * K
    meta = assets.per_line.set_index("line_index")
    for li in top_idx:
        labels[pos[li]] = (f"{meta.loc[li, 'element']} "
                           f"{meta.loc[li, 'central_wavelength_nm']:.1f}")
    for _, r in pairs.iterrows():
        qi, kj = int(r["query_line_index"]), int(r["key_line_index"])
        if qi in pos and kj in pos:
            M[pos[qi], pos[kj]] = float(r["attention"])

    fig, ax = plt.subplots(figsize=(9.8, 8.6))
    im = ax.imshow(M, cmap="viridis", aspect="equal")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=90, fontsize=8.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Key line (attended to)")
    ax.set_ylabel("Query line (attending)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_title(f"Line-line self-attention among the top {K} lines\n"
                 "(head- and layer-averaged)", fontsize=12.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Attention weight")
    reg.save(fig, "fig3b_line_pair_attention",
             f"Directed line-to-line self-attention among the {K} most "
             "important lines (rows = queries, columns = keys), ordered by "
             "attention importance.")


def _select_scatter_elements(preds, targets, element_names, n=9) -> list[int]:
    """Top-n elements by R^2 among those with meaningful correlations.

    The Spearman requirement filters out sparse trace elements whose high R^2
    comes from a few outlier samples (quantization-degenerate scatter).
    """
    from scipy.stats import spearmanr
    stats = []
    for i, _ in enumerate(element_names):
        t, p = targets[:, i], preds[:, i]
        if t.std() < 1e-9 or p.std() < 1e-9:
            continue
        pear = float(np.corrcoef(t, p)[0, 1])
        spear = spearmanr(t, p).correlation
        spear = float(spear) if np.isfinite(spear) else 0.0
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        if pear >= 0.85 and spear >= 0.5:
            stats.append((r2, i))
    stats.sort(reverse=True)
    chosen = [i for _, i in stats[:n]]
    fe = element_names.index("Fe") if "Fe" in element_names else None
    if fe is not None and fe not in chosen and chosen:
        chosen[-1] = fe
    # Order panels by R^2 but put Fe (the matrix element) first.
    if fe in chosen:
        chosen = [fe] + [i for i in chosen if i != fe]
    return chosen


def _detection_per_element_metrics(run_info: dict) -> list[tuple[str, dict]]:
    det = (run_info.get("test_results") or {}).get("detection") or {}
    per_elem = det.get("per_element") or {}
    if per_elem:
        return [(name, m) for name, m in per_elem.items()]
    return []


def _select_detection_panels(per_elem: list[tuple[str, dict]], names: list[str],
                           n: int = 9) -> list[int]:
    """Top elements by positive support (present above LOD in test set)."""
    ranked = []
    for i, name in enumerate(names):
        m = dict(per_elem.get(name, {}))
        support = float(m.get("support", 0.0))
        if support > 0:
            ranked.append((support, i))
    ranked.sort(reverse=True)
    chosen = [i for _, i in ranked[:n]]
    fe = names.index("Fe") if "Fe" in names else None
    if fe is not None and fe not in chosen and chosen:
        chosen[-1] = fe
    if fe in chosen:
        chosen = [fe] + [i for i in chosen if i != fe]
    return chosen


def draw_detection_panel(ax, y_true: np.ndarray, y_prob: np.ndarray, elem: str,
                         color: str = "#0072B2", show_xlabel: bool = True,
                         show_ylabel: bool = True):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=y_true.shape[0])
    ax.scatter(y_true + jitter, y_prob, s=7, alpha=0.25, color=color,
               edgecolors="none", rasterized=True)
    ax.axhline(0.5, color="0.4", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["absent", "present"])
    tp = int(((y_true >= 0.5) & (y_prob >= 0.5)).sum())
    fp = int(((y_true < 0.5) & (y_prob >= 0.5)).sum())
    fn = int(((y_true >= 0.5) & (y_prob < 0.5)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    ax.text(0.04, 0.96, f"{elem}\nF1 = {f1:.3f}\nP = {prec:.3f}, R = {rec:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5)
    if show_xlabel:
        ax.set_xlabel("True presence (LOD threshold)")
    if show_ylabel:
        ax.set_ylabel("Predicted probability")


def make_fig4_detection(assets: Assets, reg: FigureRegistry):
    inf = assets.inference()
    names = assets.element_names
    per_elem_raw = (assets.run_info.get("test_results") or {}).get("detection", {})
    per_elem = per_elem_raw.get("per_element") or {}
    chosen = _select_detection_panels(per_elem, names)

    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols)) if chosen else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.9 * nrows))
    axes = np.atleast_2d(axes)
    probs = inf.get("probs", inf["preds"])
    for k, ei in enumerate(chosen):
        ax = axes[k // ncols][k % ncols]
        draw_detection_panel(
            ax, inf["targets"][:, ei], probs[:, ei], names[ei],
            show_xlabel=(k // ncols == nrows - 1),
            show_ylabel=(k % ncols == 0),
        )
    for k in range(len(chosen), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Element presence detection (test set subsample)",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    reg.save(
        fig, "fig4_presence_detection",
        f"Predicted presence probability vs. LOD-derived ground truth on "
        f"{inf['preds'].shape[0]} test spectra for the {len(chosen)} "
        "most frequent elements. Dashed line = 0.5 decision threshold.",
    )

    items = _detection_per_element_metrics(assets.run_info)
    if not items:
        print("run_info has no detection per_element metrics — skipping fig4b")
        return
    items.sort(key=lambda kv: kv[1].get("f1", 0.0), reverse=True)
    names_s = [k for k, _ in items]
    vals = np.array([max(m.get("f1", 0.0), 0.0) for _, m in items])
    fig, ax = plt.subplots(figsize=(7.2, 9.5))
    colors = plt.cm.viridis(0.15 + 0.75 * vals)
    ax.barh(range(len(names_s)), vals, color=colors, edgecolor="none")
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Test F1 (element presence)")
    ax.axvline(0.9, color="0.5", lw=0.9, ls=":")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.012, 1.0), i, f"{v:.2f}", va="center", fontsize=8.2)
    ax.set_title(f"Detection performance across all {len(names_s)} elements",
                 fontsize=12.5)
    fig.tight_layout()
    reg.save(
        fig, "fig4b_per_element_f1",
        "Per-element test-set F1 for LOD-based presence/absence labels "
        "(full test split, from run_info.yaml).",
    )


def make_fig4(assets: Assets, reg: FigureRegistry):
    if assets.task == "detection":
        make_fig4_detection(assets, reg)
        return
    inf = assets.inference()
    preds, targets = inf["preds"], inf["targets"]
    names = assets.element_names
    chosen = _select_scatter_elements(preds, targets, names)

    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.9 * nrows))
    axes = np.atleast_2d(axes)
    for k, ei in enumerate(chosen):
        ax = axes[k // ncols][k % ncols]
        draw_pred_scatter(ax, targets[:, ei], preds[:, ei], names[ei],
                          show_xlabel=(k // ncols == nrows - 1),
                          show_ylabel=(k % ncols == 0))
    for k in range(len(chosen), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Predicted vs. true element concentrations (test set)",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    reg.save(fig, "fig4_pred_vs_true",
             f"Decoded concentration predictions vs. ground truth on "
             f"{preds.shape[0]} test spectra for the {len(chosen)} "
             "best-quantified elements (binned head, argmax decoding). "
             "Dashed line = identity.")

    # 4b — per-element R^2 from the full-test-set metrics in run_info.yaml.
    per_elem = (assets.run_info.get("test_results") or {}).get("per_element")
    if not per_elem:
        print("run_info has no per_element test metrics — skipping fig4b")
        return
    items = [(name, m["r2"]) for name, m in per_elem.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    names_s = [k for k, _ in items]
    vals = np.array([max(v, 0.0) for _, v in items])
    fig, ax = plt.subplots(figsize=(7.2, 9.5))
    colors = plt.cm.viridis(0.15 + 0.75 * vals)
    ax.barh(range(len(names_s)), vals, color=colors, edgecolor="none")
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Test $R^2$ (decoded concentration)")
    ax.axvline(0.9, color="0.5", lw=0.9, ls=":")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.012, 1.0), i, f"{v:.2f}", va="center", fontsize=8.2)
    ax.set_title(f"Quantification performance across all "
                 f"{len(names_s)} elements", fontsize=12.5)
    fig.tight_layout()
    reg.save(fig, "fig4b_per_element_r2",
             "Per-element test-set R^2 of decoded concentrations (full test "
             "split, from run_info.yaml); negative values clipped to 0.")


def _scalars(log_dir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    ev = ea.Scalars(tag)
    return np.array([e.value for e in ev])


def make_fig5(assets: Assets, reg: FigureRegistry):
    ft_logs = str(assets.run_dir / "logs")
    pre_run = assets.run_info.get("pretrain_run")
    pre_logs = str(Path(pre_run) / "logs") if pre_run else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    if pre_logs and Path(pre_logs).is_dir():
        tr = _scalars(pre_logs, "train/loss_epoch")
        va = _scalars(pre_logs, "val/loss")
        if tr is not None:
            ax1.plot(np.arange(1, len(tr) + 1), tr, color="#0072B2", lw=2,
                     label="train")
        if va is not None:
            ax1.plot(np.arange(1, len(va) + 1), va, color="#D55E00", lw=2,
                     label="validation")
        ax1.set_title("Self-supervised pre-training\n(masked line-intensity prediction)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "pretrain logs not found", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.axis("off")

    if assets.task == "detection":
        tr = _scalars(ft_logs, "train/det_f1_epoch")
        if tr is None:
            tr = _scalars(ft_logs, "train/det_f1")
        va = _scalars(ft_logs, "val/det_f1")
        ylab = "Detection F1"
        title = "Fine-tuning\n(element presence detection)"
        test_key = "test/det_f1"
        test_scale = 1.0
        test_fmt = "{:.3f}"
    else:
        tr = _scalars(ft_logs, "train/bin_accuracy_epoch")
        va = _scalars(ft_logs, "val/bin_accuracy")
        ylab = "Concentration-bin accuracy (%)"
        title = "Fine-tuning\n(binned element quantification)"
        test_key = "test/bin_accuracy"
        test_scale = 100.0
        test_fmt = "{:.1f}%"
    if tr is not None:
        ax2.plot(np.arange(1, len(tr) + 1), tr * test_scale if test_scale != 1.0 else tr,
                 color="#0072B2", lw=2, label="train")
    if va is not None:
        ax2.plot(np.arange(1, len(va) + 1), va * test_scale if test_scale != 1.0 else va,
                 color="#D55E00", lw=2, label="validation")
    test_acc = (assets.run_info.get("test_results") or {}).get(test_key)
    if test_acc:
        y = test_acc * test_scale if test_scale != 1.0 else test_acc
        ax2.axhline(y, color="0.4", ls=":", lw=1.4)
        ax2.text(0.98, y - (0.02 if test_scale == 1.0 else 1.2),
                 f"test {test_fmt.format(test_acc if test_scale == 1.0 else test_acc * 100)}",
                 transform=ax2.get_yaxis_transform(), ha="right", va="top",
                 fontsize=10, color="0.3")
    ax2.set_title(title)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(ylab)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    caption = (
        "Training curves parsed from TensorBoard logs: pre-training loss "
        "(left) and fine-tuning detection F1 (right), with the final test "
        "F1 marked."
        if assets.task == "detection" else
        "Training curves parsed from TensorBoard logs: pre-training loss "
        "(left) and fine-tuning concentration-bin accuracy (right), with "
        "the final test accuracy marked."
    )
    reg.save(fig, "fig5_training_curves", caption)


def make_fig6(assets: Assets, reg: FigureRegistry):
    inf = assets.inference()
    c_idx = assets.element_names.index("C")
    color_values = (
        inf["concentrations"][:, c_idx]
        if assets.task == "detection"
        else inf["targets"][:, c_idx]
    )
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    print("Computing t-SNE embedding map...")
    draw_embedding_map(ax, fig, inf["representations"], color_values,
                       seed=assets.args.seed, color_label="C content (wt.%)")
    color_label = "C content" if assets.task != "detection" else "C concentration"
    ax.set_title("Learned spectral embeddings (t-SNE)", fontsize=13)
    fig.tight_layout()
    reg.save(fig, "fig6_embedding_map",
             "t-SNE projection of the pooled encoder embeddings of test "
             f"spectra, colored by {color_label} — the model organizes spectra "
             "by composition without being given it explicitly.")


def make_fig7(assets: Assets, reg: FigureRegistry, top_n: int,
              with_inference: bool):
    lines = top_lines_table(assets, min(top_n, 12))
    elem_colors = element_color_map(list(lines["element"]))

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.34,
                          left=0.06, right=0.97, top=0.90, bottom=0.09)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    draw_annotated_spectrum(ax_a, assets, len(lines), elem_colors,
                            label_fontsize=8.5, lines_subset=lines,
                            marker_scale=0.8)
    ax_a.set_title("LIBS spectrum with model-identified significant lines",
                   fontsize=12)

    draw_element_heatmap(ax_b, assets, fig, cbar=False, tick_fontsize=9,
                         top_elements=15)
    ax_b.set_title("Element-element attention", fontsize=11.5)

    if with_inference:
        inf = assets.inference()
        c_idx = assets.element_names.index("C")
        fe_idx = assets.element_names.index("Fe")
        if assets.task == "detection":
            probs = inf.get("probs", inf["preds"])
            draw_detection_panel(
                ax_c, inf["targets"][:, fe_idx], probs[:, fe_idx], "Fe",
                color="#0072B2", show_xlabel=True, show_ylabel=True,
            )
            ax_c.set_title("Presence detection (test set)", fontsize=11.5)
            color_values = inf["concentrations"][:, c_idx]
        else:
            draw_pred_scatter(ax_c, inf["targets"][:, fe_idx], inf["preds"][:, fe_idx],
                              "Fe", color="#0072B2")
            ax_c.set_title("Quantification (test set)", fontsize=11.5)
            color_values = inf["targets"][:, c_idx]
        draw_embedding_map(ax_d, fig, inf["representations"],
                           color_values, seed=assets.args.seed,
                           cbar=True, max_points=2000,
                           color_label="C content (wt.%)")
        ax_d.set_title("Learned embeddings", fontsize=11.5)
    else:
        per_elem = (assets.run_info.get("test_results") or {}).get("per_element", {})
        items = sorted(((n, m["r2"]) for n, m in per_elem.items()),
                       key=lambda kv: kv[1], reverse=True)[:12]
        ax_c.barh(range(len(items)), [max(v, 0) for _, v in items],
                  color="#0072B2")
        ax_c.set_yticks(range(len(items)))
        ax_c.set_yticklabels([n for n, _ in items], fontsize=9)
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Test $R^2$")
        ax_c.set_title("Quantification (test set)", fontsize=11.5)
        wl_lines = assets.per_line.sort_values("line_index")
        ax_d.vlines(wl_lines["central_wavelength_nm"], 0,
                    wl_lines["importance_layer_mean"], color="0.6", lw=0.7)
        ax_d.set_xlabel("Wavelength (nm)")
        ax_d.set_ylabel("Attention")
        ax_d.set_title("Per-line importance", fontsize=11.5)

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        ax.text(-0.04, 1.12, f"({letter})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    fig.suptitle("Transformer foundation model for LIBS spectra: "
                 "interpretable line attention and element quantification",
                 fontsize=14.5, y=0.985)
    reg.save(fig, "fig7_graphical_abstract",
             "Composite graphical abstract (16:9): (a) annotated spectrum, "
             "(b) element-element attention, (c) quantification performance, "
             "(d) learned embedding map.")


def _fit_voigt_window(x: np.ndarray, y: np.ndarray, gamma_init: float,
                      sigma_init: float):
    """Voigt fit with the same setup as data.line_features.fit_line_in_spectrum,
    but returning the fitted parameters so the profile can be drawn.

    Returns (popt [x0, amplitude, gamma, sigma], r2) or (None, nan).
    """
    from scipy.optimize import curve_fit
    y = y.astype(np.float64)
    if y.size < 4 or y.max() <= 0:
        return None, float("nan")
    x0_guess = float(x[np.argmax(y)])
    y_max = float(y.max())
    lb = [float(x[0]), 0.0, 1e-4, 1e-4]
    ub = [float(x[-1]), y_max * 100.0, 0.5, 0.05]
    try:
        popt, _ = curve_fit(voigt, x, y, p0=[x0_guess, y_max, gamma_init, sigma_init],
                            bounds=(lb, ub), maxfev=2000)
    except (RuntimeError, ValueError, TypeError):
        return None, float("nan")
    if not np.all(np.isfinite(popt)) or popt[1] <= 0 or popt[3] <= 0:
        return None, float("nan")
    fit_y = voigt(x, *popt)
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return popt, r2


def make_fig8(assets: Assets, reg: FigureRegistry, n_panels: int = 3):
    """Zoomed spectral windows with the pipeline's Voigt line fits overlaid."""
    wl = assets.wavelength
    spec, _, _ = assets.spectrum_sample()
    window = assets.fit_cfg["window_nm"]
    g0, s0 = assets.fit_cfg["gamma_init"], assets.fit_cfg["sigma_init"]

    # Among the most attention-important lines, keep the best on-the-fly fits.
    candidates = top_lines_table(assets, 40)
    all_fits = []
    for _, row in candidates.iterrows():
        centre = float(row["central_wavelength_nm"])
        m = (wl >= centre - window) & (wl <= centre + window)
        if m.sum() < 6:
            continue
        popt, r2 = _fit_voigt_window(wl[m], spec[m], g0, s0)
        if popt is None or r2 < 0.85:
            continue
        all_fits.append({"element": str(row["element"]), "centre": centre,
                         "x": wl[m], "y": spec[m], "popt": popt, "r2": r2,
                         "shift": abs(float(popt[0]) - centre)})
    # Prefer clean, well-centred fits (a large centre shift means the window
    # actually caught a neighbouring line's shoulder).
    good = [f for f in all_fits if f["shift"] <= 0.05 and f["r2"] >= 0.9]
    fits = (good if len(good) >= n_panels else all_fits)[:n_panels]
    if not fits:
        print("fig8: no line window produced a valid Voigt fit — skipped")
        return

    elem_colors = element_color_map([f["element"] for f in fits])
    fig, axes = plt.subplots(1, len(fits), figsize=(4.3 * len(fits), 4.2))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fits):
        color = elem_colors.get(f["element"], ACCENT)
        x0_fit, amp, gamma, sigma = f["popt"]
        x_dense = np.linspace(f["x"][0], f["x"][-1], 400)
        ax.plot(f["x"], f["y"], "o", ms=4.5, mfc="white", mec=SPECTRUM_COLOR,
                mew=1.1, zorder=3, label="spectrum")
        ax.plot(x_dense, voigt(x_dense, *f["popt"]), color=color, lw=2.0,
                zorder=2, label="Voigt fit")
        ax.axvline(f["centre"], color="0.45", lw=1.0, ls="--", zorder=1)
        fwhm = fwhm_voigt(gamma, sigma)
        ax.set_title(f"{f['element']} {f['centre']:.2f} nm", fontsize=12.5,
                     color=color)
        ax.text(0.03, 0.96,
                f"FWHM = {fwhm * 1000:.0f} pm\n"
                f"$\\Delta\\lambda$ = {(x0_fit - f['centre']) * 1000:+.1f} pm\n"
                f"$R^2$ = {f['r2']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10)
        ax.set_xlabel("Wavelength (nm)")
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis="x", labelsize=9.5)
    axes[0].set_ylabel("Intensity (a.u.)")
    axes[-1].legend(loc="upper right", fontsize=10)
    fig.suptitle("Voigt profile fits at theoretical line centres "
                 "(token features of the model input)", fontsize=13.5, y=1.0)
    fig.tight_layout()
    reg.save(fig, "fig8_voigt_fit",
             f"Zoomed windows (±{window} nm) of the representative spectrum "
             f"around {len(fits)} attention-important lines, with the Voigt "
             "profile fitted by the tokenization pipeline (markers = spectrum "
             "samples, line = fit, dashed = theoretical centre). Fitted FWHM, "
             "centre shift and R^2 annotated; these fits provide the "
             "max-intensity/FWHM token features the model consumes.")


# ────────────────────────────────────────────────────────────────────────────
# GIFs
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _per_layer_importance(assets: Assets, n_samples: int, batch_size: int):
    """[n_layers, n_lines] mean CLS attention per layer over test spectra."""
    encoder = assets.encoder
    device = assets.device
    rng = np.random.default_rng(assets.args.seed)
    test_idx = assets.splits["test"]
    sub = np.sort(rng.choice(test_idx, size=min(n_samples, len(test_idx)),
                             replace=False))
    n_layers = len(encoder.encoder_blocks)
    n_lines = assets.token_meta["n_lines"]
    sums = np.zeros((n_layers, n_lines), dtype=np.float64)
    n_seen = 0
    print(f"Collecting per-layer attention over {len(sub)} spectra...")
    with h5py.File(assets.tokens_path, "r") as f:
        for start in range(0, len(sub), batch_size):
            idx = sub[start:start + batch_size]
            batch = {
                "tokens": torch.from_numpy(f["tokens"][idx].astype(np.float32)),
                "fit_valid": torch.from_numpy(f["fit_valid"][idx].astype(np.uint8)),
            }
            batch, _ = _drop_zero_valid_spectra(batch)
            if batch is None:
                continue
            inputs = {"tokens": batch["tokens"].to(device),
                      "fit_valid": batch["fit_valid"].to(device)}
            cls_rows, kpm = cls_attention_per_layer(encoder, inputs)
            for li, row in enumerate(cls_rows):
                norm = _normalize_line_attention([row], kpm)["layer_mean"]
                sums[li] += norm.sum(dim=0).cpu().numpy()
            n_seen += inputs["tokens"].shape[0]
    return sums / max(n_seen, 1), n_seen


def make_gif_layers(assets: Assets, reg: FigureRegistry, fps_ms: int = 1200):
    per_layer, n_seen = _per_layer_importance(
        assets, n_samples=assets.args.attn_samples, batch_size=4,
    )
    n_layers = per_layer.shape[0]
    line_wl = (assets.per_line.sort_values("line_index")
               ["central_wavelength_nm"].to_numpy())
    meta = assets.per_line.sort_values("line_index").reset_index(drop=True)
    _, mean_spec, _ = assets.spectrum_sample()
    wl = assets.wavelength
    ymax_imp = per_layer.max() * 1.25
    spec_norm = mean_spec / mean_spec.max()

    frames = []
    for li in range(n_layers):
        fig, ax = plt.subplots(figsize=(11, 4.8), dpi=110)
        ax.plot(wl, spec_norm * ymax_imp * 0.55, lw=0.6, color="0.78", zorder=1)
        imp = per_layer[li]
        ax.vlines(line_wl, 0, imp, color="#0072B2", lw=1.0, zorder=2)
        top5 = np.argsort(imp)[::-1][:5]
        xs = line_wl[top5].astype(np.float64)
        x0, x1 = wl.min() - 5, wl.max() + 5
        # Horizontal labels are ~8 characters wide -> need generous spacing.
        min_dx = (x1 - x0) * 0.085
        label_x = spread_positions(xs, x0 + min_dx, x1 - min_dx, min_dx)
        for k, t in enumerate(top5):
            elem = meta.loc[t, "element"]
            ax.plot(line_wl[t], imp[t], "o", ms=4, color=ACCENT, zorder=3)
            ax.plot([line_wl[t], label_x[k]],
                    [imp[t] + 0.015 * ymax_imp, ymax_imp * 0.88],
                    color=ACCENT, lw=0.6, alpha=0.55, zorder=2)
            ax.text(label_x[k], ymax_imp * 0.90, f"{elem} {line_wl[t]:.1f}",
                    ha="center", va="bottom", fontsize=9.5, color=ACCENT)
        ax.set_ylim(0, ymax_imp)
        ax.set_xlim(wl.min() - 5, wl.max() + 5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("CLS attention")
        ax.set_title(f"Where the model looks — transformer layer {li + 1} / "
                     f"{n_layers}   (mean over {n_seen} test spectra)")
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    out = reg.output_dir / "anim_attention_layers.gif"
    durations = [fps_ms] * (len(frames) - 1) + [fps_ms * 2]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    reg.add_file(out.name,
                 "Animation of the CLS-token attention across the "
                 f"{n_layers} transformer layers (gray = mean spectrum, "
                 "blue = attention per line, top-5 lines labeled per layer).")
    print(f"Saved: {out.name}")


def make_gif_buildup(assets: Assets, reg: FigureRegistry, n_lines: int = 20,
                     frame_ms: int = 700):
    lines = top_lines_table(assets, n_lines)
    elem_colors = element_color_map(list(lines["element"]))
    wl = assets.wavelength
    spec, _, _ = assets.spectrum_sample()
    ymax = float(spec.max())

    frames = []
    for k in range(0, n_lines + 1):
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=110)
        ax.plot(wl, spec, lw=0.7, color=SPECTRUM_COLOR, zorder=2)
        annotate_lines(ax, lines, wl, spec, elem_colors, ymax,
                       k_visible=k, fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        title = ("LIBS spectrum" if k == 0 else
                 f"Top {k} spectral lines by model attention")
        ax.set_title(title, pad=14)
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    out = reg.output_dir / "anim_line_buildup.gif"
    durations = [1500] + [frame_ms] * (len(frames) - 2) + [4000]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    reg.add_file(out.name,
                 f"Animation revealing the top {n_lines} attention-important "
                 "lines one by one (descending importance) on a representative "
                 "test spectrum.")
    print(f"Saved: {out.name}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main(args):
    run_dir = Path(args.run_dir)
    if not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"not a run directory: {run_dir}")

    targets = ALL_TARGETS if not args.only else [
        t.strip() for t in args.only.split(",") if t.strip()
    ]
    unknown = [t for t in targets if t not in ALL_TARGETS]
    if unknown:
        raise ValueError(f"unknown --only targets {unknown}; pick from {ALL_TARGETS}")
    if args.skip_inference:
        skipped = [t for t in targets if t in ("fig4", "fig6", "gif_layers")]
        targets = [t for t in targets if t not in ("fig4", "fig6", "gif_layers")]
        if skipped:
            print(f"--skip-inference: skipping {skipped} "
                  "(fig4b is also skipped; it is bundled with fig4)")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = run_dir / "evaluation" / f"publication_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    plt.rcParams.update(PUB_RC)
    assets = Assets(run_dir, args)
    reg = FigureRegistry(output_dir)

    try:
        if "fig1" in targets:
            make_fig1(assets, reg, args.top_lines)
        if "fig2" in targets:
            make_fig2(assets, reg, args.top_lines)
        if "fig3" in targets:
            make_fig3(assets, reg)
        if "fig4" in targets:
            make_fig4(assets, reg)
        if "fig5" in targets:
            make_fig5(assets, reg)
        if "fig6" in targets:
            make_fig6(assets, reg)
        if "fig7" in targets:
            make_fig7(assets, reg, args.top_lines,
                      with_inference=not args.skip_inference)
        if "fig8" in targets:
            make_fig8(assets, reg)
        if "gif_buildup" in targets:
            make_gif_buildup(assets, reg)
        if "gif_layers" in targets:
            make_gif_layers(assets, reg)
    finally:
        assets.close()

    reg.write_readme([
        f"Run: {assets.run_info.get('run_name', run_dir.name)}",
        f"Task: {assets.run_info.get('task')}  "
        f"Embedding: {assets.run_info.get('embedding_type')}",
        f"Attention source: {assets.attention_dir.name}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "All static figures: 300 dpi PNG + SVG with editable text "
        "(white background, PowerPoint-ready).",
    ])

    print("\n" + "=" * 60)
    print(f"Publication figures complete: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render publication-quality figures + GIFs for a "
                    "fine-tuned LIBS foundation-model run",
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN,
                        help="Fine-tuned run directory (binned quantification, "
                             "line_token_linear embedding)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write figures (default: "
                             "<run_dir>/evaluation/publication_<timestamp>)")
    parser.add_argument("--max_samples", type=int, default=4096,
                        help="Test spectra used for checkpoint inference "
                             "(fig4 scatter, fig6 embeddings)")
    parser.add_argument("--attn_samples", type=int, default=64,
                        help="Test spectra aggregated for the per-layer "
                             "attention GIF")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--top_lines", type=int, default=15,
                        help="Number of annotated lines in spectrum figures")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-inference", dest="skip_inference",
                        action="store_true",
                        help="Skip checkpoint inference (drops fig4, fig6, "
                             "gif_layers; abstract uses CSV-only panels)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma list of targets to render: "
                             + ",".join(ALL_TARGETS))
    main(parser.parse_args())
