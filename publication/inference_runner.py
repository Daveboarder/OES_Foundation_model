"""
Embedding-aware checkpoint inference for publication / comparison figures.

Supports intensity (raw wavelength bins) and line_token_linear runs on the
same test-split indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from analyze_attention_importance import _checkpoint_encoder_state, build_encoder
from data.line_tokenization import FEATURE_NAMES
from models.heads import concentration_to_presence
from training.finetune import LIBSFinetuneModule
from utils.run_manager import RunManager


def resolve_cache_path(recorded: str | None, cache_dir: Path, pattern: str) -> Path:
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


def resolve_spectra_cache(cache_dir: Path, n_total: int) -> Path:
    for pattern in ("synthetic_cache_*.h5", "measured_cache_*.h5"):
        for cand in sorted(cache_dir.glob(pattern)):
            with h5py.File(cand, "r") as f:
                if f["spectra"].shape[0] == n_total:
                    return cand
    return resolve_cache_path(None, cache_dir, "synthetic_cache_*.h5")


def load_splits(cache_dir: Path, n_total: int, n_test: int) -> dict[str, np.ndarray]:
    for cand in sorted(cache_dir.glob("splits_*.json")):
        s = json.load(open(cand))
        if len(s.get("test", [])) == n_test and sum(len(v) for v in s.values()) == n_total:
            return {k: np.asarray(v, dtype=np.int64) for k, v in s.items()}
    raise FileNotFoundError(
        f"no splits_*.json matching n_total={n_total}, n_test={n_test} in {cache_dir}",
    )


def filter_indices_with_valid_tokens(tokens_path: Path, indices: np.ndarray) -> np.ndarray:
    """Keep only spectra that have at least one valid Voigt fit (token path)."""
    with h5py.File(tokens_path, "r") as f:
        valid = f["fit_valid"]
        keep = [int(i) for i in indices if valid[int(i)].sum() > 0]
    return np.asarray(keep, dtype=np.int64)


class FinetuneInferenceRunner:
    """Run detection inference for one fine-tuned run."""

    def __init__(
        self,
        run_dir: str | Path,
        device: str = "auto",
        batch_size: int = 32,
        label: str = "",
    ):
        self.run_dir = Path(run_dir)
        self.label = label or self.run_dir.name
        self.batch_size = batch_size
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.config = yaml.safe_load(open(self.run_dir / "config.yaml"))
        self.run_info = yaml.safe_load(open(self.run_dir / "run_info.yaml"))
        self.element_names: list[str] = list(self.run_info["element_names"])
        self.embedding_type = str(
            self.run_info.get("embedding_type")
            or self.config.get("model", {}).get("embedding_type", "intensity"),
        )
        self.cache_dir = Path("external_data/cache")
        self._module: LIBSFinetuneModule | None = None
        self._token_meta: dict | None = None
        self._spectra_path: Path | None = None
        self._tokens_path: Path | None = None
        self._concentrations: np.ndarray | None = None
        self._lod: torch.Tensor | None = None

    @property
    def n_total_spectra(self) -> int:
        return (
            self.run_info["train_samples"]
            + self.run_info["val_samples"]
            + self.run_info["test_samples"]
        )

    @property
    def splits(self) -> dict[str, np.ndarray]:
        return load_splits(
            self.cache_dir,
            self.n_total_spectra,
            self.run_info["test_samples"],
        )

    @property
    def spectra_path(self) -> Path:
        if self._spectra_path is None:
            self._spectra_path = resolve_spectra_cache(self.cache_dir, self.n_total_spectra)
        return self._spectra_path

    @property
    def tokens_path(self) -> Path:
        if self._tokens_path is None:
            self._tokens_path = resolve_cache_path(
                self.run_info.get("line_tokens_path"),
                self.cache_dir,
                "line_tokens_*.h5",
            )
        return self._tokens_path

    @property
    def token_meta(self) -> dict | None:
        if self.embedding_type != "line_token_linear":
            return None
        if self._token_meta is None:
            with h5py.File(self.tokens_path, "r") as f:
                self._token_meta = {
                    "n_lines": int(f.attrs["n_lines"]),
                    "n_features": int(f.attrs["n_features"]),
                    "feature_names": FEATURE_NAMES,
                    "feature_mean": np.asarray(f.attrs["feature_mean"], dtype=np.float32),
                    "feature_std": np.asarray(f.attrs["feature_std"], dtype=np.float32),
                    "central_wavelength": f["central_wavelength"][:].astype(np.float32),
                }
        return self._token_meta

    @property
    def concentrations(self) -> np.ndarray:
        if self._concentrations is None:
            with h5py.File(self.spectra_path, "r") as f:
                g = f["sample_table"]
                cols = [
                    np.asarray(g[name], dtype=np.float32) for name in self.element_names
                ]
                conc = np.stack(cols, axis=1)
            np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            self._concentrations = np.clip(conc, 0.0, 1.0)
        return self._concentrations

    @property
    def lod_vector(self) -> torch.Tensor:
        if self._lod is None:
            lod_map = self.run_info.get("element_lod") or {}
            default = float(self.run_info.get("default_lod", 1e-4))
            lod = np.array(
                [float(lod_map.get(name, default)) for name in self.element_names],
                dtype=np.float32,
            )
            self._lod = torch.from_numpy(lod)
        return self._lod

    def finetune_checkpoint(self) -> Path:
        best = self.run_dir / "checkpoints" / "best.ckpt"
        if best.is_file():
            return best
        ckpt = RunManager.from_existing_run(str(self.run_dir)).get_checkpoint_for_mode("finetune")
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint in {self.run_dir}")
        return Path(ckpt)

    @property
    def module(self) -> LIBSFinetuneModule:
        if self._module is not None:
            return self._module

        cfg = yaml.safe_load(open(self.run_dir / "config.yaml"))
        token_meta = self.token_meta
        if self.embedding_type == "line_token_linear":
            if token_meta is None:
                raise ValueError("line_token_linear run requires token cache")
            cfg["data"]["n_bins"] = token_meta["n_lines"]
            cfg["model"]["max_seq_len"] = token_meta["n_lines"] + 1
        else:
            with h5py.File(self.spectra_path, "r") as f:
                n_bins = int(f["spectra"].shape[1])
            cfg["data"]["n_bins"] = n_bins
            cfg["model"]["max_seq_len"] = n_bins + 1

        encoder = build_encoder(cfg, self.run_info, token_meta)
        ckpt = self.finetune_checkpoint()
        state = _checkpoint_encoder_state(str(ckpt))
        enc_sd = encoder.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in enc_sd and enc_sd[k].shape == v.shape
        }
        encoder.load_state_dict(filtered, strict=False)

        module = LIBSFinetuneModule(
            encoder=encoder,
            task=self.run_info["task"],
            n_classes=cfg["data"]["n_classes"],
            n_elements=self.run_info["n_elements"],
            n_concentration_bins=self.run_info["n_concentration_bins"],
            pool=self.run_info["pool"],
            element_names=self.element_names,
            lod=self.lod_vector,
        )
        ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj
        mod_sd = module.state_dict()
        filtered_mod = {
            k: v for k, v in sd.items()
            if k in mod_sd and mod_sd[k].shape == v.shape
        }
        module.load_state_dict(filtered_mod, strict=False)
        self._module = module.to(self.device).eval()
        print(f"[{self.label}] Loaded module from {ckpt.name} ({self.embedding_type})")
        return self._module

    @torch.no_grad()
    def run_detection_inference(self, indices: np.ndarray) -> dict:
        """Detection outputs on the given global spectrum indices."""
        indices = np.asarray(indices, dtype=np.int64)
        module = self.module
        device = self.device
        lod = self.lod_vector.to(device)
        conc_all = self.concentrations[indices]

        preds, probs, reprs, targets, concs = [], [], [], [], []
        bs = self.batch_size
        if self.embedding_type == "intensity":
            bs = min(bs, 4)
        print(f"[{self.label}] Inference on {len(indices)} spectra ({device}, bs={bs})...")

        if self.embedding_type == "line_token_linear":
            with h5py.File(self.tokens_path, "r") as f:
                tok_ds, valid_ds = f["tokens"], f["fit_valid"]
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
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
                        conc_all[start:start + bs][keep.numpy()].astype(np.float32),
                    ).to(device)
                    preds.append(out["presence_pred"].cpu().numpy())
                    probs.append(out["presence_prob"].cpu().numpy())
                    targets.append(concentration_to_presence(conc_batch, lod).cpu().numpy())
                    reprs.append(out["representation"].float().cpu().numpy())
                    concs.append(conc_batch.cpu().numpy())
        else:
            with h5py.File(self.spectra_path, "r") as f:
                spec_ds = f["spectra"]
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
                    spectra = torch.from_numpy(
                        spec_ds[idx].astype(np.float32),
                    ).to(device)
                    batch = {"spectrum": spectra}
                    out = module(batch)
                    conc_batch = torch.from_numpy(
                        conc_all[start:start + bs].astype(np.float32),
                    ).to(device)
                    preds.append(out["presence_pred"].cpu().numpy())
                    probs.append(out["presence_prob"].cpu().numpy())
                    targets.append(concentration_to_presence(conc_batch, lod).cpu().numpy())
                    reprs.append(out["representation"].float().cpu().numpy())
                    concs.append(conc_batch.cpu().numpy())

        return {
            "indices": indices,
            "preds": np.concatenate(preds, axis=0),
            "probs": np.concatenate(probs, axis=0),
            "targets": np.concatenate(targets, axis=0),
            "concentrations": np.concatenate(concs, axis=0),
            "representations": np.concatenate(reprs, axis=0),
        }


def load_pretrain_summary(pretrain_run: str | None) -> dict:
    if not pretrain_run:
        return {}
    path = Path(pretrain_run) / "run_info.yaml"
    if not path.is_file():
        return {"path": pretrain_run, "error": "run_info not found"}
    info = yaml.safe_load(open(path))
    return {
        "path": pretrain_run,
        "name": info.get("run_name", Path(pretrain_run).name),
        "experiment": info.get("experiment_name"),
        "embedding_type": info.get("embedding_type"),
        "pretrain_loss": info.get("pretrain_loss", "mse"),
        "epochs": info.get("epochs"),
        "n_lines": info.get("n_lines"),
        "best_val_loss": info.get("best_val_loss"),
    }
