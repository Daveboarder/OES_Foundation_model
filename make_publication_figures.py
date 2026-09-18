"""
Publication-quality figures for the LIBS foundation model.

Reads the data caches + outputs of a fine-tuned `quantification_binned`,
`detection` or `cf_quantification` run (line_token_linear embedding) and
renders a PowerPoint-ready figure set: 300 dpi PNG + editable-text SVG, white
background, large fonts.

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

Task-specific variants of fig4 (detection: `fig4_presence_detection` +
`fig4b_per_element_f1`; cf_quantification: `fig4_cf_pred_vs_true` (log-log,
censored points marked) + `fig4b_cf_per_element_within2x`).

Calibration-free (cf_quantification) extras
    fig_cf_sb_plot               Saha-Boltzmann plots of 3 test spectra
    fig_cf_plasma_recovery       T and log10 Ne predicted vs true (one-/two-zone)
    fig_cf_comparison            CF-learned vs CF pure-physics vs binned seed bars

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

    # CF run: add a pure-physics run to the comparison figure
    uv run python make_publication_figures.py --run_dir runs/finetune_<cf> \
        --compare_runs "CF (pure physics)=runs/finetune_<cf_pure>"
"""

from __future__ import annotations

import argparse
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
from data.line_tokenization import (  # noqa: E402
    F_EK, F_ION, F_LOG_AK, F_LOG_GK, F_MAX_I, F_WAVELENGTH, F_Z, FEATURE_NAMES,
)
from models.heads import bin_to_concentration, concentration_to_presence  # noqa: E402
from publication.inference_runner import (  # noqa: E402
    CF_TASK,
    FinetuneInferenceRunner,
    load_splits,
    resolve_spectra_cache,
)
from training.finetune import LIBSFinetuneModule  # noqa: E402
from utils.run_manager import RunManager  # noqa: E402

DEFAULT_RUN = "runs/finetune_2026-06-04_21-03-15_libs_binned_ft"

CF_TARGETS = ["fig_cf_sb_plot", "fig_cf_plasma_recovery", "fig_cf_comparison"]
ALL_TARGETS = [
    "fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
    "gif_layers", "gif_buildup",
] + CF_TARGETS
# Targets that need checkpoint inference (dropped by --skip-inference).
INFERENCE_TARGETS = ("fig4", "fig6", "gif_layers", "fig_cf_sb_plot",
                     "fig_cf_plasma_recovery")

# Major elements of the Fe-matrix sample set (order = panel/bar order).
CF_MAJOR_ELEMENTS = ["Fe", "C", "Mn", "Si", "Cr", "Ni", "Cu", "Al"]

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

# Calibration-free figures follow the dataviz reference palette: categorical
# slots are assigned in this FIXED order (never cycled past 8 — extra classes
# fold into "other"), text/axes use ink tokens, never a series colour.
CF_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CF_INK = "#0b0b0b"
CF_INK_SECONDARY = "#52514e"
CF_MUTED = "#898781"       # axis labels, "other", censored markers
CF_HAIRLINE = "#e1e0d9"    # gridlines
CF_BASELINE = "#c3c2b7"    # identity / reference lines
# Method colours for the comparison figure: colour follows the method, not
# its position in the bar group, so every panel paints a method identically.
CF_METHOD_SLOTS = {
    "CF (learned)": CF_PALETTE[0],
    "CF (pure physics)": CF_PALETTE[1],
    "Binned seed": CF_PALETTE[2],
    "Classical (54 lines)": CF_PALETTE[3],
}
CF_ZONE_COLORS = {"one-zone": CF_PALETTE[0], "two-zone": CF_PALETTE[1]}


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
        self.cf_info: dict = dict(self.run_info.get("cf") or {})
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
        self._cf_runner: FinetuneInferenceRunner | None = None
        self._plasma_targets: dict[str, np.ndarray] | None = None

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
    def is_cf(self) -> bool:
        return self.task == CF_TASK

    @property
    def spectra_h5(self) -> h5py.File:
        """Spectra cache of the run: `synthetic_cache_*.h5` or
        `measured_cache_*.h5`, preferring the path recorded in run_info."""
        if self._spectra_file is None:
            path = resolve_spectra_cache(self.cache_dir, self.n_total_spectra,
                                         self.run_info)
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
            self._splits = load_splits(
                self.cache_dir, self.n_total_spectra, self.run_info["test_samples"],
                strategy=self.cf_info.get("split_strategy"),
            )
        return self._splits

    # ── calibration-free run helpers ──
    @property
    def cf_runner(self) -> FinetuneInferenceRunner:
        """Checkpoint runner that knows how to rebuild a CF module
        (tables, layer config and seeds from run_info['cf'])."""
        if self._cf_runner is None:
            self._cf_runner = FinetuneInferenceRunner(
                self.run_dir, device=self.args.device,
                batch_size=self.args.batch_size, label=self.run_dir.name,
            )
        return self._cf_runner

    @property
    def plasma_targets(self) -> dict[str, np.ndarray]:
        """Contract C2 aux targets (Te, log10_Ne, log10_Nl, is_two_zone,
        has_plasma_labels) for every spectrum of the cache."""
        if self._plasma_targets is None:
            self._plasma_targets = self.cf_runner.plasma_targets
        return self._plasma_targets

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
        if self._encoder is None and self.is_cf:
            # The CF module owns the encoder (built with its tables/seeds).
            self._encoder = self.cf_runner.module.encoder
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
        """Full finetune module (encoder + task head) loaded from best.ckpt."""
        if self._module is None and self.is_cf:
            self._module = self.cf_runner.module
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

        if self.is_cf:
            # Physics solver outputs (mass fractions, T, Ne, per-line weights…)
            # come from the runner; keys follow run_cf_inference's contract.
            print(f"Running CF inference on {n_take} test spectra "
                  f"(batch_size={self.args.batch_size}, device={self.device})...")
            self._inference = self.cf_runner.run_cf_inference(sub)
            return self._inference

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
    if n < 5:
        # smoke runs have a handful of test spectra; t-SNE needs more than that
        ax.text(0.5, 0.5, f"t-SNE needs ≥ 5 spectra (got {n})",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(reprs.shape[0], size=n, replace=False)
    X = reprs[pick].astype(np.float64)
    emb = TSNE(n_components=2, random_state=seed,
               perplexity=float(max(2.0, min(30.0, (n - 1) / 3.0))),
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
    if assets.is_cf:
        make_fig4_cf(assets, reg)
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


# ────────────────────────────────────────────────────────────────────────────
# Calibration-free (cf_quantification) figures
# ────────────────────────────────────────────────────────────────────────────

def _scatter_alpha(n: int) -> float:
    """Point opacity that keeps sparse scatters readable and dense ones light."""
    return 0.8 if n < 200 else (0.5 if n < 1000 else 0.35)


def _cf_style_axes(ax):
    """Recessive chrome for CF figures: hairline grid, muted axis ink."""
    ax.grid(True, color=CF_HAIRLINE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(CF_BASELINE)
    ax.tick_params(colors=CF_INK_SECONDARY)


def _cf_log_metrics(y_true: np.ndarray, y_pred: np.ndarray, censored: np.ndarray,
                    lod: float) -> tuple[float, float, int, int]:
    """(log-RMSE, within-2x fraction, n scored, n censored) over uncensored
    predictions of spectra whose true content is at/above the LOD."""
    scored = (y_true >= lod) & ~censored & (y_pred > 0)
    n_cens = int((censored & (y_true >= lod)).sum())
    if scored.sum() == 0:
        return float("nan"), float("nan"), 0, n_cens
    d = np.log(y_pred[scored]) - np.log(y_true[scored])
    return (float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d) <= np.log(2.0))),
            int(scored.sum()), n_cens)


def draw_cf_scatter(ax, y_true: np.ndarray, y_pred: np.ndarray, elem: str,
                    lod: float, censored: np.ndarray | None = None,
                    color: str = CF_PALETTE[0], show_xlabel: bool = True,
                    show_ylabel: bool = True):
    """Log–log CF prediction vs truth for one element.

    Filled dots = solver estimates; hollow triangles (muted) = predictions the
    solver flagged as censored (below LOD or no usable line). True values of
    zero cannot be drawn on a log axis and are dropped; predictions below
    LOD/10 are clamped to that floor so censored points stay visible.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    censored = (np.zeros_like(y_true, dtype=bool) if censored is None
                else np.asarray(censored, dtype=bool))
    keep = y_true > 0
    t, p, c = y_true[keep], y_pred[keep], censored[keep]
    floor = max(lod / 10.0, 1e-8)
    p_draw = np.maximum(p, floor)
    scale = 100.0  # wt.%
    lo = min(t.min(), p_draw.min()) * scale * 0.6 if t.size else floor * scale
    hi = max(t.max(), p_draw.max()) * scale * 1.6 if t.size else 1.0

    line_x = np.array([lo, hi])
    ax.plot(line_x, line_x, color=CF_BASELINE, lw=1.0, zorder=1)
    ax.plot(line_x, 2.0 * line_x, color=CF_BASELINE, lw=0.8, ls="--", zorder=1)
    ax.plot(line_x, 0.5 * line_x, color=CF_BASELINE, lw=0.8, ls="--", zorder=1)
    ax.axvline(lod * scale, color=CF_MUTED, lw=0.8, ls=":", zorder=1)

    ok = ~c
    ax.scatter(t[ok] * scale, p_draw[ok] * scale, s=10, alpha=_scatter_alpha(int(ok.sum())),
               color=color, edgecolors="none", zorder=3, rasterized=True)
    if c.any():
        ax.scatter(t[c] * scale, p_draw[c] * scale, s=16, marker="v",
                   facecolors="none", edgecolors=CF_MUTED, linewidths=0.7,
                   alpha=0.8, zorder=2, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    _cf_style_axes(ax)

    log_rmse, within2x, n_scored, n_cens = _cf_log_metrics(t, p, c, lod)
    ax.text(0.04, 0.96,
            f"{elem}\nlog-RMSE = {log_rmse:.2f}\nwithin 2x = {within2x:.0%}\n"
            f"n = {n_scored}" + (f", censored {n_cens}" if n_cens else ""),
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=CF_INK)
    if show_xlabel:
        ax.set_xlabel("True (wt.%)")
    if show_ylabel:
        ax.set_ylabel("CF predicted (wt.%)")
    return log_rmse, within2x


def _select_cf_panels(targets: np.ndarray, lod: np.ndarray, names: list[str],
                      n: int = 9, min_support: int = 5) -> list[int]:
    """Elements with the most test spectra above LOD; Fe (matrix) first."""
    support = (targets >= lod[None, :]).sum(axis=0)
    ranked = sorted(((int(support[i]), i) for i in range(len(names))
                     if support[i] >= min_support), reverse=True)
    chosen = [i for _, i in ranked[:n]]
    fe = names.index("Fe") if "Fe" in names else None
    if fe is not None and fe not in chosen and chosen:
        chosen[-1] = fe
    if fe in chosen:
        chosen = [fe] + [i for i in chosen if i != fe]
    return chosen


def _per_element_metric_items(per_elem: dict, metric: str) -> list[tuple[str, float]]:
    """(element, value) pairs for `metric`, tolerating one nested level
    (e.g. per_element[el][sample_median][metric])."""
    items: list[tuple[str, float]] = []
    for name, m in (per_elem or {}).items():
        if not isinstance(m, dict):
            continue
        val = m.get(metric)
        if val is None:
            for sub in m.values():
                if isinstance(sub, dict) and sub.get(metric) is not None:
                    val = sub[metric]
                    break
        if val is not None and np.isfinite(float(val)):
            items.append((str(name), float(val)))
    return items


def make_fig4_cf(assets: Assets, reg: FigureRegistry):
    inf = assets.inference()
    preds, targets = inf["preds"], inf["targets"]
    censored = inf.get("censored")
    names = assets.element_names
    lod = assets.lod_vector.numpy()
    chosen = _select_cf_panels(targets, lod, names)
    if not chosen:
        print("fig4 (cf): no element with enough test support — skipped")
        return

    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.9 * nrows))
    axes = np.atleast_2d(axes)
    for k, ei in enumerate(chosen):
        ax = axes[k // ncols][k % ncols]
        draw_cf_scatter(ax, targets[:, ei], preds[:, ei], names[ei], float(lod[ei]),
                        censored=None if censored is None else censored[:, ei],
                        show_xlabel=(k // ncols == nrows - 1),
                        show_ylabel=(k % ncols == 0))
    for k in range(len(chosen), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    variant = "pure physics" if assets.cf_info.get("pure_physics") else "learned weights"
    fig.suptitle(f"Calibration-free quantification vs. true composition "
                 f"(test set, {variant})", fontsize=14, y=1.0)
    fig.tight_layout()
    n_cens_total = int(censored[:, chosen].sum()) if censored is not None else 0
    reg.save(fig, "fig4_cf_pred_vs_true",
             f"Calibration-free (Saha–Boltzmann + closure) mass fractions vs. "
             f"ground truth on {preds.shape[0]} test spectra for the {len(chosen)} "
             "elements with the largest test support, log–log axes. Solid line = "
             "identity, dashed = factor-of-two band, dotted = LOD; hollow "
             f"triangles = censored predictions ({n_cens_total} in these panels). "
             "Annotated log-RMSE and within-2x fraction are computed on "
             "uncensored spectra with true content at or above LOD.")

    # 4b — per-element within-2x from the full-test-set metrics in run_info.
    per_elem = (assets.run_info.get("test_results") or {}).get("per_element")
    items = _per_element_metric_items(per_elem, "within_2x")
    if not items:
        print("run_info has no per_element within_2x metrics — skipping fig4b")
        return
    n_cens = dict(_per_element_metric_items(per_elem, "n_censored"))
    items.sort(key=lambda kv: kv[1], reverse=True)
    names_s = [k for k, _ in items]
    vals = np.array([min(max(v, 0.0), 1.0) for _, v in items])
    fig, ax = plt.subplots(figsize=(7.2, 9.5))
    ax.barh(range(len(names_s)), vals, color=CF_PALETTE[0], edgecolor="none",
            height=0.72, zorder=2)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Test fraction within a factor of 2 (uncensored, above LOD)")
    _cf_style_axes(ax)
    ax.grid(False, axis="y")
    for i, (name, v) in enumerate(zip(names_s, vals)):
        label = f"{v:.2f}"
        if name in n_cens and n_cens[name] > 0:
            label += f"  ({int(n_cens[name])} censored)"
        ax.text(min(v + 0.012, 1.0), i, label, va="center", fontsize=8.2,
                color=CF_INK_SECONDARY)
    ax.set_title(f"Calibration-free accuracy across all {len(names_s)} elements",
                 fontsize=12.5)
    fig.tight_layout()
    reg.save(fig, "fig4b_cf_per_element_within2x",
             "Per-element fraction of test spectra whose calibration-free "
             "estimate lies within a factor of two of the truth (full test "
             "split, from run_info.yaml); the number of censored predictions "
             "(below LOD / no usable line) is given in brackets.")


def _cf_sb_points(tokens: np.ndarray, valid: np.ndarray, weights: np.ndarray,
                  T: float, log10_Ne: float, tables, tau0: np.ndarray | None,
                  gamma_nm: float, used_mask: np.ndarray | None = None) -> dict:
    """Saha–Boltzmann coordinates of every used line of one spectrum.

    x = E_k + z·E_ion,   y = ln(area·λ / (g_k A_k)) − z·ln F(T) + z·ln N_e,
    so that all lines of one element (both stages) lie on y = q_e − x / kT.
    Areas are self-absorption corrected with the solver's curve-of-growth
    factor when `tau0` is available (raw positions are kept in `y_raw`).
    """
    from data.plasma_physics import (
        KB_EV, curve_of_growth_factor, doppler_sigma_nm, saha_thermal_factor,
    )
    wl = tokens[:, F_WAVELENGTH].astype(np.float64)
    Ek = tokens[:, F_EK].astype(np.float64)
    gk = 10.0 ** tokens[:, F_LOG_GK].astype(np.float64)
    Ak = 10.0 ** tokens[:, F_LOG_AK].astype(np.float64)
    Z = np.rint(tokens[:, F_Z]).astype(np.int64)
    z = np.rint(tokens[:, F_ION]).astype(np.int64)
    area = tokens[:, F_MAX_I].astype(np.float64)
    elem_idx = np.full(Z.shape, -1, dtype=np.int64)
    in_range = (Z >= 0) & (Z < len(tables.z_to_elem))
    elem_idx[in_range] = np.asarray(tables.z_to_elem)[Z[in_range]]

    used = (valid > 0) & (weights > 1e-3) & (area > 0) & (elem_idx >= 0)
    if used_mask is not None:
        used &= used_mask.astype(bool)
    e = elem_idx[used]
    x = Ek[used] + z[used] * np.asarray(tables.E_ion)[e]
    lnF = float(np.log(saha_thermal_factor(T)))
    eta = float(np.log(10.0) * log10_Ne)
    y_raw = np.log(area[used] * wl[used] / (gk[used] * Ak[used])) - z[used] * lnF + z[used] * eta
    y = y_raw.copy()
    tau = None
    if tau0 is not None:
        tau = np.maximum(np.asarray(tau0, dtype=np.float64)[used], 0.0)
        sigma = doppler_sigma_nm(wl[used], T, np.asarray(tables.mass_amu)[e])
        f = curve_of_growth_factor(tau, sigma, gamma_nm)
        y = y_raw - np.log(np.maximum(f, 1e-12))
    return {"x": x, "y": y, "y_raw": y_raw, "elem": e, "w": weights[used],
            "z": z[used], "tau0": tau, "beta": 1.0 / (KB_EV * T)}


def make_fig_cf_sb_plot(assets: Assets, reg: FigureRegistry, n_spectra: int = 3):
    """Saha–Boltzmann plots of three test spectra spanning the fitted T range."""
    if not assets.is_cf:
        print("fig_cf_sb_plot: not a cf_quantification run — skipped")
        return
    inf = assets.inference()
    if inf.get("weights") is None or inf.get("T") is None:
        print("fig_cf_sb_plot: module output lacks cf_weights/cf_T — skipped")
        return
    tables = assets.cf_runner.cf_tables
    names = assets.element_names
    weights, valid = inf["weights"], inf["fit_valid"]
    used_all = inf.get("used_mask")
    n_used = ((weights > 1e-3) & (valid > 0)).sum(axis=1)
    if used_all is not None:
        n_used = np.minimum(n_used, used_all.sum(axis=1))
    # Candidates: the better-populated third of the spectra; pick min/median/max T.
    order = np.argsort(-n_used)
    pool = order[:max(n_spectra, len(order) // 3)]
    pool = pool[np.argsort(inf["T"][pool])]
    if len(pool) >= n_spectra:
        picks = [int(pool[0]), int(pool[len(pool) // 2]), int(pool[-1])][:n_spectra]
        picks = list(dict.fromkeys(picks))
    else:
        picks = [int(i) for i in pool]
    if not picks:
        print("fig_cf_sb_plot: no spectrum with used lines — skipped")
        return

    cf_cfg = dict(assets.cf_info.get("cf_cfg") or {})
    gamma_nm = float(cf_cfg.get("gamma_nm", 0.01))
    sa_on = bool(cf_cfg.get("sa_correction", True)) and inf.get("tau0") is not None
    glob_idx = inf["indices"][picks]
    with h5py.File(assets.tokens_path, "r") as f:
        tokens = f["tokens"][np.sort(glob_idx)].astype(np.float32)
    # h5py needs sorted indices; restore pick order.
    tokens = tokens[np.argsort(np.argsort(glob_idx))]

    panels = []
    for k, row in enumerate(picks):
        panels.append(_cf_sb_points(
            tokens[k], valid[row], weights[row], float(inf["T"][row]),
            float(inf["log10_Ne"][row]), tables,
            inf["tau0"][row] if sa_on else None, gamma_nm,
            used_mask=None if used_all is None else used_all[row],
        ))

    # Element colours: fixed slots by total weight over the shown spectra;
    # everything past slot 7 folds into a muted "other".
    tot_w = np.zeros(len(names))
    for p in panels:
        np.add.at(tot_w, p["elem"], p["w"])
    ranked = [int(i) for i in np.argsort(-tot_w) if tot_w[i] > 0]
    slots = {ei: CF_PALETTE[r] for r, ei in enumerate(ranked[:7])}
    other_color = CF_MUTED

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) + 0.8, 4.9),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, p, row in zip(axes, panels, picks):
        T, logNe = float(inf["T"][row]), float(inf["log10_Ne"][row])
        size = 6.0 + 54.0 * np.clip(p["w"], 0.0, 1.0)
        for ei in np.unique(p["elem"]):
            m = p["elem"] == ei
            color = slots.get(int(ei), other_color)
            ax.scatter(p["x"][m], p["y"][m], s=size[m], color=color, alpha=0.75,
                       edgecolors="white", linewidths=0.5, zorder=3)
            if sa_on and p["tau0"] is not None:
                strong = m & (p["tau0"] > 0.5)
                if strong.any():
                    ax.scatter(p["x"][strong], p["y_raw"][strong], s=size[strong] * 0.6,
                               facecolors="none", edgecolors=color, linewidths=0.6,
                               alpha=0.6, zorder=2)
                    ax.vlines(p["x"][strong], p["y_raw"][strong], p["y"][strong],
                              color=color, lw=0.5, alpha=0.5, zorder=2)
            # Common-slope line from the returned intercept q_e and T.
            q = inf["intercepts"][row, int(ei)] if inf.get("intercepts") is not None else np.nan
            if np.isfinite(q) and int(ei) in slots:
                xs = np.array([p["x"][m].min() - 0.3, p["x"][m].max() + 0.3])
                ax.plot(xs, q - p["beta"] * xs, color=color, lw=1.4, alpha=0.9, zorder=2)
        _cf_style_axes(ax)
        ax.set_xlabel("$E_k + z\\,E_{ion}$ (eV)")
        ax.set_title(f"T = {T:.0f} K,  log$_{{10}}$ N$_e$ = {logNe:.2f}\n"
                     f"{int(len(p['x']))} weighted lines", fontsize=11.5)
    axes[0].set_ylabel("ln(A λ / g$_k$ A$_{ki}$) − z ln F(T) + z ln N$_e$")

    handles = [plt.Line2D([], [], marker="o", ls="", color=slots[ei], ms=7,
                          label=names[ei]) for ei in ranked[:7]]
    if len(ranked) > 7:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=other_color, ms=7,
                                  label="other"))
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=CF_INK_SECONDARY, ms=3.5,
                   label="w = 0.25"),
        plt.Line2D([], [], marker="o", ls="", color=CF_INK_SECONDARY, ms=7.5,
                   label="w = 1.0"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=9, ncol=1,
                    handletextpad=0.4, borderaxespad=0.2)
    fig.suptitle("Saha–Boltzmann plots of calibration-free test spectra "
                 "(points sized by learned line weight)", fontsize=13.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    reg.save(fig, "fig_cf_sb_plot",
             f"Saha–Boltzmann plots for {len(panels)} test spectra spanning the "
             "fitted temperature range. Each point is one weighted line "
             "(x = upper-level energy plus z·E_ion, y = ln(area·λ/(g_k A_ki)) "
             "minus the Saha thermal term plus z·ln N_e, so neutral and ionic "
             "lines of one element share a line); point size encodes the "
             "reliability weight, colour the element (top 7 by weight, rest "
             "muted). Solid lines have the common slope −1/kT from the fitted "
             "temperature and the per-element intercept q_e returned by the "
             "solver" + (". Hollow markers show the uncorrected position of "
                        "self-absorbed lines (τ0 > 0.5)." if sa_on else "."))


def _draw_recovery_panel(ax, true_v, pred_v, two_zone, xlabel, ylabel,
                         fmt_err, title):
    """Predicted-vs-true scatter coloured one-/two-zone with per-group error."""
    lo = float(min(true_v.min(), pred_v.min()))
    hi = float(max(true_v.max(), pred_v.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=CF_BASELINE, lw=1.0,
            zorder=1)
    lines = []
    for label, m in (("one-zone", ~two_zone), ("two-zone", two_zone)):
        if not m.any():
            continue
        ax.scatter(true_v[m], pred_v[m], s=10, alpha=_scatter_alpha(int(two_zone.size)),
                   color=CF_ZONE_COLORS[label], edgecolors="none", zorder=3,
                   rasterized=True, label=label)
        lines.append(f"{label}: {fmt_err(true_v[m], pred_v[m])} (n={int(m.sum())})")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    _cf_style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11.5)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", fontsize=9.5, color=CF_INK)


def make_fig_cf_plasma_recovery(assets: Assets, reg: FigureRegistry):
    """T and log10 Ne recovered by the CF solver vs the generator's values."""
    if not assets.is_cf:
        print("fig_cf_plasma_recovery: not a cf_quantification run — skipped")
        return
    inf = assets.inference()
    if inf.get("T") is None or inf.get("log10_Ne") is None:
        print("fig_cf_plasma_recovery: module output lacks cf_T/cf_log10_Ne — skipped")
        return
    aux = assets.plasma_targets
    idx = inf["indices"]
    has = aux["has_plasma_labels"][idx] > 0
    if not has.any():
        print("fig_cf_plasma_recovery: spectra cache has no plasma labels "
              "(measured data or legacy generator) — skipped")
        return
    two_zone = aux["is_two_zone"][idx][has] > 0
    T_true, T_pred = aux["Te"][idx][has], inf["T"][has]
    ne_true, ne_pred = aux["log10_Ne"][idx][has], inf["log10_Ne"][has]

    def mape(t, p):
        return f"MAPE {np.mean(np.abs(p - t) / np.maximum(t, 1.0)):.1%}"

    def mae(t, p):
        return f"MAE {np.mean(np.abs(p - t)):.2f} dex"

    panels = [
        (T_true / 1000.0, T_pred / 1000.0, "True T$_e$ (kK)", "CF T (kK)", mape,
         "Temperature"),
        (ne_true, ne_pred, "True log$_{10}$ N$_e$ (cm$^{-3}$)",
         "CF log$_{10}$ N$_e$", mae, "Electron density"),
    ]
    nl_true = aux.get("log10_Nl")
    nl_init = inf.get("init_log10_Nl0")
    if nl_true is not None and nl_init is not None and np.any(nl_true[idx][has] != 0):
        panels.append((nl_true[idx][has], nl_init[has],
                       "True log$_{10}$ (N·l) (cm$^{-2}$)",
                       "Initial-guess log$_{10}$ (N·l)", mae, "Column density (init head)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels) + 0.4, 4.6))
    axes = np.atleast_1d(axes)
    for ax, (t, p, xl, yl, err, title) in zip(axes, panels):
        _draw_recovery_panel(ax, np.asarray(t, dtype=np.float64),
                             np.asarray(p, dtype=np.float64), two_zone, xl, yl, err, title)
    axes[0].legend(loc="lower right", fontsize=9.5, markerscale=2.0)
    fig.suptitle("Plasma-state recovery by the calibration-free solver (test set)",
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    reg.save(fig, "fig_cf_plasma_recovery",
             f"Plasma parameters recovered by the Saha–Boltzmann solver on "
             f"{int(has.sum())} synthetic test spectra vs. the generator's values "
             "(inner-zone T_e1 and N_e1 for two-zone shots): temperature, "
             "electron density" + (", and the initial column-density guess of "
                                   "the plasma-init head" if len(panels) == 3 else "")
             + ". Colour marks one-zone vs. two-zone shots; a single-zone "
             "solver applied to a two-zone plasma recovers an effective "
             "temperature, hence the larger two-zone scatter.")


def _find_per_element_blocks(d, path: tuple[str, ...] = (), depth: int = 0):
    """Yield (path, per_element_dict) for every nested dict carrying a
    `per_element` block (test_results and evaluate_cf.py layouts)."""
    if not isinstance(d, dict) or depth > 4:
        return
    pe = d.get("per_element")
    if isinstance(pe, dict) and pe and all(isinstance(v, dict) for v in pe.values()):
        yield path, pe
    for k, v in d.items():
        if k != "per_element" and isinstance(v, dict):
            yield from _find_per_element_blocks(v, path + (str(k),), depth + 1)


def _cf_method_label(path: tuple[str, ...], default: str) -> str:
    s = "/".join(path).lower()
    if "pure" in s:
        return "CF (pure physics)"
    if "classical" in s:
        return "Classical (54 lines)"
    if "binned" in s or "seed" in s:
        return "Binned seed"
    return default


def _cf_group_label(path: tuple[str, ...], default: str) -> str:
    return "measured" if "measured" in "/".join(path).lower() else default


def _cf_comparison_sources(assets: Assets) -> dict[tuple[str, str], dict]:
    """{(group, method): per_element} from this run, its binned seed run and
    any `--compare_runs label=run_dir` entries. Groups: 'synthetic test' and
    'measured' (blocks under test_results_measured)."""
    sources: dict[tuple[str, str], dict] = {}

    def add_run(run_info: dict, default_label: str):
        for top, group in (("test_results", "synthetic test"),
                           ("test_results_measured", "measured")):
            block = run_info.get(top)
            if not isinstance(block, dict):
                continue
            for path, pe in _find_per_element_blocks(block, (top,)):
                key = (_cf_group_label(path, group), _cf_method_label(path[1:], default_label))
                sources.setdefault(key, pe)

    own_label = ("CF (pure physics)" if assets.cf_info.get("pure_physics")
                 else "CF (learned)")
    add_run(assets.run_info, own_label)

    seed_run = assets.cf_info.get("seed_binned_run")
    if seed_run and Path(seed_run, "run_info.yaml").is_file():
        add_run(yaml.safe_load(open(Path(seed_run, "run_info.yaml"))) or {}, "Binned seed")

    for entry in (assets.args.compare_runs or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, _, run_dir = entry.partition("=")
        if not run_dir:
            label, run_dir = Path(entry).name, entry
        info_path = Path(run_dir) / "run_info.yaml"
        if not info_path.is_file():
            print(f"--compare_runs: {run_dir} has no run_info.yaml — skipped")
            continue
        add_run(yaml.safe_load(open(info_path)) or {}, label.strip())
    return sources


def make_fig_cf_comparison(assets: Assets, reg: FigureRegistry):
    """Grouped bars per major element: CF-learned vs CF pure-physics vs the
    binned seed (and any extra runs), on the synthetic test split and, when
    `test_results_measured` exists, on measured spectra."""
    if not assets.is_cf:
        print("fig_cf_comparison: not a cf_quantification run — skipped")
        return
    sources = _cf_comparison_sources(assets)
    if not sources:
        print("fig_cf_comparison: no per_element test metrics found — skipped")
        return
    names = assets.element_names
    majors = [e for e in CF_MAJOR_ELEMENTS if e in names]

    # Metric: the one available in the most sources (priority on ties).
    candidates = [("within_2x", "Fraction within a factor of 2", False),
                  ("r2", "Test $R^2$", False),
                  ("log_rmse", "log-RMSE (lower is better)", True)]
    cf_keys = {k for k in sources if k[1].startswith("CF")}
    counts = {}
    for metric, _label, _lb in candidates:
        have = [k for k, pe in sources.items()
                if any(e in dict(_per_element_metric_items(pe, metric)) for e in majors)]
        counts[metric] = (sum(1 for k in have if k in cf_keys), len(have))
    # The figure is about CF: take the first metric the CF runs themselves
    # report, never r2 just because a seed run has it — r2 in linear wt.% is
    # meaningless for a solver judged in log space (it clips to an empty bar).
    best = next((c for c in candidates if counts[c[0]][0] > 0), None)
    if best is None:  # no CF metric at all: fall back to the widest-reported one
        best = max(candidates, key=lambda c: counts[c[0]][1])
    metric, metric_label, lower_better = best
    n_have = counts[metric][1]
    if n_have == 0:
        print("fig_cf_comparison: no comparable per-element metric — skipped")
        return

    groups = [g for g in ("synthetic test", "measured") if any(k[0] == g for k in sources)]
    method_order = list(CF_METHOD_SLOTS) + sorted(
        {k[1] for k in sources} - set(CF_METHOD_SLOTS))
    methods = [m for m in method_order if any(k[1] == m for k in sources)]
    colors = {m: CF_METHOD_SLOTS.get(m, CF_PALETTE[min(4 + i, 7)])
              for i, m in enumerate(methods)}

    panel_w = 0.8 * len(majors) + 1.8
    fig, axes = plt.subplots(1, len(groups), figsize=(panel_w * len(groups), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(majors))
    ymax = 0.0
    for ax, group in zip(axes, groups):
        present = [m for m in methods if (group, m) in sources]
        for j, m in enumerate(present):
            vals = dict(_per_element_metric_items(sources[(group, m)], metric))
            y = np.array([vals.get(e, np.nan) for e in majors], dtype=np.float64)
            if not lower_better:
                y = np.clip(y, 0.0, None)
            offset = (j - (len(present) - 1) / 2.0) * width
            ax.bar(x + offset, np.nan_to_num(y), width * 0.92, color=colors[m],
                   edgecolor="none", label=m, zorder=2)
            for xi, yi in zip(x + offset, y):
                if np.isnan(yi):
                    ax.text(xi, 0.01, "n/a", rotation=90, ha="center", va="bottom",
                            fontsize=7, color=CF_MUTED)
            ymax = max(ymax, float(np.nanmax(y)) if np.isfinite(y).any() else 0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(majors)
        ax.set_title(group, fontsize=12)
        _cf_style_axes(ax)
        ax.grid(False, axis="x")
    axes[0].set_ylabel(metric_label)
    if not lower_better:
        axes[0].set_ylim(0, 1.12)
    else:
        axes[0].set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    # One legend for every method drawn anywhere (colour follows the method).
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[m], edgecolor="none", label=m) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), fontsize=10,
               bbox_to_anchor=(0.5, 0.96), frameon=False)
    fig.suptitle("Calibration-free vs. learned quantification per major element",
                 fontsize=13.5, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    reg.save(fig, "fig_cf_comparison",
             f"Per-element {metric_label.replace('$', '')} for the major elements "
             f"({', '.join(majors)}) of the methods available in run_info.yaml "
             f"({', '.join(methods)}) on the synthetic test split"
             + (" and on measured spectra (test_results_measured)"
                if "measured" in groups else "")
             + ". Missing bars (n/a) mean the metric was not reported for that "
             "method/element.")


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

    legend_loc = "lower right"
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
    elif assets.is_cf:
        tr = _scalars(ft_logs, "train/cf_log_rmse_epoch")
        if tr is None:
            tr = _scalars(ft_logs, "train/cf_log_rmse")
        va = _scalars(ft_logs, "val/cf_log_rmse")
        ylab = "CF log-RMSE (ln units, lower is better)"
        title = "Fine-tuning\n(calibration-free quantification: line weights + plasma init)"
        test_key = "test/cf_log_rmse"
        test_scale = 1.0
        test_fmt = "{:.3f}"
        legend_loc = "upper right"
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
    if tr is None and va is None:
        ax2.text(0.5, 0.5, "fine-tune curves not found", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title(title)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(ylab)
    if tr is not None or va is not None:
        ax2.legend(loc=legend_loc)
    fig.tight_layout()
    if assets.task == "detection":
        caption = (
            "Training curves parsed from TensorBoard logs: pre-training loss "
            "(left) and fine-tuning detection F1 (right), with the final test "
            "F1 marked."
        )
    elif assets.is_cf:
        caption = (
            "Training curves parsed from TensorBoard logs: pre-training loss "
            "(left) and the calibration-free log-RMSE of the solver output "
            "while the line-weight and plasma-init heads are trained on "
            "synthetic data (right), with the final test log-RMSE marked."
        )
    else:
        caption = (
            "Training curves parsed from TensorBoard logs: pre-training loss "
            "(left) and fine-tuning concentration-bin accuracy (right), with "
            "the final test accuracy marked."
        )
    reg.save(fig, "fig5_training_curves", caption)


def make_fig6(assets: Assets, reg: FigureRegistry):
    inf = assets.inference()
    c_idx = assets.element_names.index("C")
    if assets.task == "detection":
        color_values = inf["concentrations"][:, c_idx]
    elif assets.is_cf:
        # CF targets are the true mass fractions of the spectra cache.
        color_values = inf["targets"][:, c_idx]
    else:
        color_values = inf["targets"][:, c_idx]
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
        elif assets.is_cf:
            cens = inf.get("censored")
            draw_cf_scatter(ax_c, inf["targets"][:, fe_idx], inf["preds"][:, fe_idx],
                            "Fe", float(assets.lod_vector[fe_idx]),
                            censored=None if cens is None else cens[:, fe_idx])
            ax_c.set_title("Calibration-free quantification (test set)", fontsize=11.5)
            color_values = inf["targets"][:, c_idx]
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
        metric, xlabel = "r2", "Test $R^2$"
        if assets.is_cf and _per_element_metric_items(per_elem, "within_2x"):
            metric, xlabel = "within_2x", "Test fraction within 2x"
        items = sorted(_per_element_metric_items(per_elem, metric),
                       key=lambda kv: kv[1], reverse=True)[:12]
        ax_c.barh(range(len(items)), [max(v, 0) for _, v in items],
                  color=CF_PALETTE[0] if assets.is_cf else "#0072B2")
        ax_c.set_yticks(range(len(items)))
        ax_c.set_yticklabels([n for n, _ in items], fontsize=9)
        ax_c.invert_yaxis()
        ax_c.set_xlabel(xlabel)
        ax_c.set_title("Calibration-free quantification (test set)"
                       if assets.is_cf else "Quantification (test set)",
                       fontsize=11.5)
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
        skipped = [t for t in targets if t in INFERENCE_TARGETS]
        targets = [t for t in targets if t not in INFERENCE_TARGETS]
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
    if not assets.is_cf:
        # CF-only figures are silently dropped for detection / binned runs
        # unless they were requested explicitly.
        implicit_cf = [t for t in targets if t in CF_TARGETS] if not args.only else []
        targets = [t for t in targets if t not in implicit_cf]

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
        if "fig_cf_sb_plot" in targets:
            make_fig_cf_sb_plot(assets, reg)
        if "fig_cf_plasma_recovery" in targets:
            make_fig_cf_plasma_recovery(assets, reg)
        if "fig_cf_comparison" in targets:
            make_fig_cf_comparison(assets, reg)
    finally:
        assets.close()

    header = [
        f"Run: {assets.run_info.get('run_name', run_dir.name)}",
        f"Task: {assets.run_info.get('task')}  "
        f"Embedding: {assets.run_info.get('embedding_type')}",
        f"Attention source: {assets.attention_dir.name}",
    ]
    if assets.is_cf:
        header.append(
            f"CF: pure_physics={assets.cf_info.get('pure_physics', False)}  "
            f"c0_source={assets.cf_info.get('c0_source')}  "
            f"seed_binned_run={assets.cf_info.get('seed_binned_run')}  "
            f"seed_detection_run={assets.cf_info.get('seed_detection_run')}",
        )
    header += [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "All static figures: 300 dpi PNG + SVG with editable text "
        "(white background, PowerPoint-ready).",
    ]
    reg.write_readme(header)

    print("\n" + "=" * 60)
    print(f"Publication figures complete: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render publication-quality figures + GIFs for a "
                    "fine-tuned LIBS foundation-model run",
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN,
                        help="Fine-tuned run directory (quantification_binned, "
                             "detection or cf_quantification; line_token_linear "
                             "embedding)")
    parser.add_argument("--compare_runs", type=str, default=None,
                        help="cf_quantification only: comma list of "
                             "'label=run_dir' entries whose run_info test metrics "
                             "are added to fig_cf_comparison (e.g. a "
                             "--cf_pure_physics run or the binned seed run)")
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
                             "gif_layers, fig_cf_sb_plot, fig_cf_plasma_recovery; "
                             "abstract uses CSV-only panels)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma list of targets to render: "
                             + ",".join(ALL_TARGETS))
    main(parser.parse_args())
