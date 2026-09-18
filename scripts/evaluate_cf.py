"""
Zero-shot calibration-free (CF) evaluation of a `cf_quantification` run on
measured spectra (or any other libs-data config).

Rebuilds the dataset and the line-token cache exactly like train_finetune.py
(cache hits when scripts/build_line_tokens.py already ran), loads the CF
module the same way the training test pass does (encoder + CF heads + frozen
seeds + Saha–Boltzmann layer), runs inference on the requested indices and
writes to ``<run_dir>/evaluation/cf_measured_<timestamp>/``:

    per_spectrum.csv     one row per spectrum: plasma state, per-element
                         prediction / truth / censored flag, binned-seed prediction
    per_element.yaml     spectrum-level and per-sample-median metrics per element
                         (mae, log_rmse, within_2x, r2, n, n_censored) for the CF
                         solver and, side by side, for the binned seed
    per_instrument.yaml  the same CF metrics grouped by instrument
                         (measured_groups(sample_table, by='instrument'))

and appends ``test_results_measured`` to ``<run_dir>/run_info.yaml``.

Usage:
    uv run python scripts/evaluate_cf.py --run_dir runs/finetune_<cf> \
        --libs_data_config config/libs_data_measured.yaml \
        [--line_embedding_config config/line_embedding_cf.yaml] \
        [--pure_physics] [--indices all|test] [--batch_size 32] [--device auto]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_attention_importance import _checkpoint_encoder_state, build_encoder  # noqa: E402
from data.dataset import LineTokensLabeledDataset  # noqa: E402
from data.libs_pipeline import (  # noqa: E402
    build_dataset_from_config,
    extract_finetune_labels,
    get_or_make_splits,
)
from data.line_embedding_pipeline import prepare_line_tokens_assets  # noqa: E402
from train_finetune import (  # noqa: E402
    SPLIT_STRATEGIES,
    align_model_section_with_pretrain,
    build_cf_assets,
    finetune_checkpoint_path,
    groups_for_strategy,
    load_module_state_shape_safe,
    plasma_targets_for_table,
)
from training.finetune import CF_MAJOR_ELEMENTS, LIBSFinetuneModule  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def pick_device(requested: str) -> str:
    """'auto' → cuda when it is available *and* allocatable (the 4090 runs in
    exclusive mode, so a busy device falls back to cpu with a warning)."""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return "cuda"
    except RuntimeError as exc:
        print(f"WARNING: CUDA unavailable ({str(exc).splitlines()[0]}); using cpu")
        return "cpu"


def concentrations_for_elements(sample_table: pd.DataFrame, element_names: list[str]) -> np.ndarray:
    """[N, E] mass fractions in the run's element order; columns missing from
    the table (never in the sample matrix) are zero with a warning."""
    present = [e for e in element_names if e in sample_table.columns]
    missing = [e for e in element_names if e not in sample_table.columns]
    conc = np.zeros((len(sample_table), len(element_names)), dtype=np.float32)
    if present:
        sub, _, _ = extract_finetune_labels(sample_table, elements=present)
        for j, e in enumerate(present):
            conc[:, element_names.index(e)] = sub[:, j]
    if missing:
        print(f"WARNING: {len(missing)} run elements absent from the sample table "
              f"(treated as 0): {missing}")
    return conc


def element_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lod: float,
    censored: np.ndarray | None,
    eps: float,
) -> dict[str, float]:
    """mae / r2 / n plus the CF log metrics (shared with the training test pass)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    cen = None if censored is None else np.asarray(censored, dtype=np.float64)[ok]
    out: dict[str, float] = {"n": float(y_true.size)}
    if y_true.size == 0:
        out.update({"mae": float("nan"), "r2": float("nan"), "log_rmse": float("nan"),
                    "within_2x": float("nan"), "n_censored": 0.0, "n_uncensored_truth": 0.0,
                    "lod": float(lod)})
        return out
    out["mae"] = float(np.mean(np.abs(y_pred - y_true)))
    out.update(LIBSFinetuneModule.cf_element_metrics(y_true, y_pred, lod, censored=cen, eps=eps))
    # R² only where the element is actually present (above LOD) in at least two
    # spectra; otherwise ss_tot is numerical noise and R² is meaningless.
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    spread = float(np.ptp(y_true)) > 1e-6 * max(float(np.max(np.abs(y_true))), 1e-12)
    out["r2"] = (float(1.0 - ss_res / ss_tot)
                 if (spread and ss_tot > 0 and out["n_uncensored_truth"] >= 2) else float("nan"))
    return out


def per_sample_median(values: np.ndarray, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median of `values` [N, E] within each sample id → ([S, E], sample ids [S])."""
    df = pd.DataFrame(values)
    df["_sid"] = sample_ids
    med = df.groupby("_sid", sort=True).median(numeric_only=True)
    return med.to_numpy(dtype=np.float64), med.index.to_numpy().astype(str)


def per_element_table(
    element_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lod: np.ndarray,
    censored: np.ndarray | None,
    sample_ids: np.ndarray,
    eps: float,
) -> dict[str, dict]:
    """{element: {spectrum_level: {...}, sample_median: {...}}} + macro summary."""
    med_true, sids = per_sample_median(y_true, sample_ids)
    med_pred, _ = per_sample_median(y_pred, sample_ids)
    med_cen = None
    if censored is not None:
        med_cen, _ = per_sample_median(censored.astype(np.float64), sample_ids)
        med_cen = med_cen >= 0.5
    out: dict[str, dict] = {}
    for i, name in enumerate(element_names):
        out[name] = {
            "spectrum_level": element_block(
                y_true[:, i], y_pred[:, i], float(lod[i]),
                None if censored is None else censored[:, i], eps),
            "sample_median": element_block(
                med_true[:, i], med_pred[:, i], float(lod[i]),
                None if med_cen is None else med_cen[:, i], eps),
        }
    return out


def macro_summary(table: dict[str, dict], element_names: list[str], level: str) -> dict[str, float]:
    """Unweighted means over elements that have uncensored truth; majors separately."""
    def _mean(names: list[str], key: str) -> float:
        vals = [table[n][level][key] for n in names
                if n in table and table[n][level].get("n_uncensored_truth", 0) > 0
                and np.isfinite(table[n][level].get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    majors = [e for e in CF_MAJOR_ELEMENTS if e in element_names]
    return {
        "log_rmse_macro": _mean(element_names, "log_rmse"),
        "within_2x_macro": _mean(element_names, "within_2x"),
        "r2_macro": _mean(element_names, "r2"),
        "log_rmse_major": _mean(majors, "log_rmse"),
        "within_2x_major": _mean(majors, "within_2x"),
        "r2_major": _mean(majors, "r2"),
        "n_elements_scored": float(sum(
            1 for n in element_names if n in table and table[n][level].get("n_uncensored_truth", 0) > 0)),
    }


def _py(obj):
    """yaml-safe: numpy scalars → python, NaN kept as float('nan')."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# module loading
# ─────────────────────────────────────────────────────────────────────────────
def load_cf_module(
    run_dir: Path,
    run_info: dict,
    config: dict,
    element_names: list[str],
    libs_data_config: str,
    token_meta: dict,
    pure_physics: bool,
    spectra_cache_path: str | None,
    split_strategy: str | None,
) -> tuple[LIBSFinetuneModule, dict]:
    """The CF module exactly as train_finetune's test pass builds it, on the
    current token cache (seeds checked by line-dictionary hash, not by cache
    basename, because the evaluation cache differs from the training cache)."""
    cf_info = dict(run_info.get("cf") or {})
    if run_info.get("task") != "cf_quantification":
        raise ValueError(f"{run_dir} is a {run_info.get('task')!r} run, not cf_quantification")
    n_lines = int(token_meta["n_lines"])
    config["data"]["n_bins"] = n_lines
    config["model"]["max_seq_len"] = n_lines + 1
    config["model"]["embedding_type"] = run_info.get("embedding_type", "line_token_linear")
    config.setdefault("finetune", {})["cf"] = dict(cf_info.get("cf_cfg") or {})

    encoder = build_encoder(config, run_info, token_meta)
    ckpt = finetune_checkpoint_path(run_dir)
    enc_state = _checkpoint_encoder_state(str(ckpt))
    enc_sd = encoder.state_dict()
    encoder.load_state_dict(
        {k: v for k, v in enc_state.items() if k in enc_sd and enc_sd[k].shape == v.shape},
        strict=False,
    )

    seed_args = SimpleNamespace(
        seed_binned_run_dir=cf_info.get("seed_binned_run"),
        seed_detection_run_dir=cf_info.get("seed_detection_run"),
        cf_pure_physics=bool(pure_physics or cf_info.get("pure_physics", False)),
        cf_c0_source=str(cf_info.get("c0_source", "binned")),
        element_lod_config=run_info.get("element_lod_config") or "config/element_lod.yaml",
    )
    assets = build_cf_assets(
        seed_args, config, element_names, libs_data_config, token_meta,
        spectra_cache_path=spectra_cache_path, split_strategy=split_strategy,
        strict_tokens=False,
    )
    module = LIBSFinetuneModule(
        encoder=encoder,
        task="cf_quantification",
        n_classes=int(config["data"].get("n_classes", 10)),
        n_elements=len(element_names),
        n_concentration_bins=int(run_info.get("n_concentration_bins", 1000)),
        pool=run_info.get("pool", "cls"),
        element_names=element_names,
        cf_tables=assets["cf_tables"],
        cf_cfg=assets["cf_cfg"],
        seed_binned=assets["seed_binned"],
        seed_detection=assets["seed_detection"],
    )
    load_module_state_shape_safe(module, ckpt)
    module.eval()
    print(f"Loaded CF module from {ckpt} (pure_physics={seed_args.cf_pure_physics}, "
          f"c0_source={seed_args.cf_c0_source})")
    return module, assets


# ─────────────────────────────────────────────────────────────────────────────
# inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    module: LIBSFinetuneModule,
    dataset: LineTokensLabeledDataset,
    batch_size: int,
    device: str,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """CF (+ binned seed) outputs for every item of `dataset`; spectra without a
    single valid Voigt fit are skipped (NaN rows) like the publication runner."""
    module = module.to(device)
    E = module.n_elements
    n = len(dataset)
    out = {
        "pred": np.full((n, E), np.nan, dtype=np.float64),
        "censored": np.zeros((n, E), dtype=bool),
        "n_lines_used": np.full((n, E), np.nan, dtype=np.float64),
        "cf_T": np.full(n, np.nan), "cf_log10_Ne": np.full(n, np.nan),
        "T0": np.full(n, np.nan), "log10_Ne0": np.full(n, np.nan), "log10_Nl0": np.full(n, np.nan),
        "mean_weight": np.full(n, np.nan), "n_valid_lines": np.zeros(n, dtype=np.int64),
        "binned_pred": np.full((n, E), np.nan, dtype=np.float64),
        "c0": np.full((n, E), np.nan, dtype=np.float64),
        "skipped": np.zeros(n, dtype=bool),
    }
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    t0 = time.time()
    pos = 0
    for bi, batch in enumerate(loader):
        B = batch["tokens"].shape[0]
        valid = batch["fit_valid"]
        n_valid = valid.sum(dim=1).numpy()
        keep = n_valid > 0
        out["n_valid_lines"][pos:pos + B] = n_valid
        out["skipped"][pos:pos + B] = ~keep
        if keep.any():
            sub = {k: (v[keep] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            sub = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sub.items()}
            res = module(sub)
            rows = np.flatnonzero(keep) + pos
            out["pred"][rows] = res["concentrations_pred"].detach().float().cpu().numpy()
            out["censored"][rows] = res["cf_censored"].detach().cpu().numpy().astype(bool)
            out["n_lines_used"][rows] = res["cf_n_lines_used"].detach().float().cpu().numpy()
            out["cf_T"][rows] = res["cf_T"].detach().float().cpu().numpy().reshape(-1)
            out["cf_log10_Ne"][rows] = res["cf_log10_Ne"].detach().float().cpu().numpy().reshape(-1)
            for key in ("T0", "log10_Ne0", "log10_Nl0"):
                out[key][rows] = res["cf_init"][key].detach().float().cpu().numpy().reshape(-1)
            w = res["cf_weights"].detach().float()
            fv = res["cf_fit_valid"].detach().float()
            out["mean_weight"][rows] = ((w * fv).sum(1) / fv.sum(1).clamp(min=1)).cpu().numpy()
            if "cf_C0" in res:
                out["c0"][rows] = res["cf_C0"].detach().float().cpu().numpy()
            if module.seed_binned is not None:
                sb = module.seed_binned({"tokens": sub["tokens"], "fit_valid": sub["fit_valid"]})
                out["binned_pred"][rows] = sb["concentrations_pred"].detach().float().cpu().numpy()
        pos += B
        if (bi + 1) % 20 == 0 or pos == n:
            print(f"  {pos}/{n} spectra ({time.time() - t0:.0f} s)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_info = yaml.safe_load(open(run_dir / "run_info.yaml"))
    config = yaml.safe_load(open(run_dir / "config.yaml"))
    align_model_section_with_pretrain(config, run_info.get("pretrain_run"))
    element_names = list(run_info["element_names"])
    line_embedding_config = args.line_embedding_config or run_info.get("line_embedding_config")
    if not line_embedding_config:
        raise ValueError("run_info has no line_embedding_config; pass --line_embedding_config")

    # 1. dataset + tokens (same path as train_finetune → cache hits)
    libs_cfg = yaml.safe_load(open(args.libs_data_config))
    libs_cfg.setdefault("generation", {})
    ds = build_dataset_from_config(libs_cfg)
    if len(ds) == 0:
        raise RuntimeError(f"{args.libs_data_config} produced no spectra")
    token_meta = prepare_line_tokens_assets(
        ds.spectra.astype(np.float32), ds.wavelength, line_embedding_config,
        spectra_cache_key=ds.cache_key, verbose=True,
    )
    tokens_path = token_meta["line_tokens_path"]
    spectra_cache_path = str(ds._cache_path()) if hasattr(ds, "_cache_path") else None
    print(f"Spectra: {len(ds)}  tokens: {Path(tokens_path).name} ({token_meta['n_lines']} lines)")

    conc_all = concentrations_for_elements(ds.sample_table, element_names)
    aux_all = plasma_targets_for_table(ds.sample_table)
    sample_ids_all = ds.sample_table["sample_type_id"].astype(str).to_numpy()
    unique_ids_all = (ds.sample_table["unique_id"].astype(str).to_numpy()
                      if "unique_id" in ds.sample_table.columns else sample_ids_all)
    try:
        instruments_all = groups_for_strategy(ds.sample_table, "group_instrument")
    except Exception as exc:  # synthetic tables have no unique_id → one group
        print(f"WARNING: no instrument grouping ({exc}); using a single group")
        instruments_all = np.asarray(["all"] * len(ds), dtype=str)

    # 2. indices
    downstream = libs_cfg.get("downstream", {})
    split_cfg = downstream.get("splits", {})
    strategy = str(args.split_strategy or split_cfg.get("strategy", "random"))
    if args.indices == "all":
        indices = np.arange(len(ds), dtype=np.int64)
    else:
        kw = dict(n=len(ds), cache_dir=ds.cache_dir, cache_key=ds.cache_key,
                  val_fraction=split_cfg.get("val_fraction", 0.15),
                  test_fraction=split_cfg.get("test_fraction", 0.15),
                  seed=split_cfg.get("seed", 42))
        if strategy != "random":
            groups = groups_for_strategy(ds.sample_table, strategy)
            splits, _ = get_or_make_splits(**kw, groups=groups, strategy=strategy)
        else:
            splits, _ = get_or_make_splits(**kw)
        indices = np.asarray(splits["test"], dtype=np.int64)
    print(f"Evaluating {len(indices)} spectra (--indices {args.indices}, split strategy {strategy})")

    # 3. module
    device = pick_device(args.device)
    module, assets = load_cf_module(
        run_dir, run_info, config, element_names, args.libs_data_config, token_meta,
        pure_physics=args.pure_physics, spectra_cache_path=spectra_cache_path,
        split_strategy=strategy,
    )
    lod = np.asarray(assets["cf_tables"].lod, dtype=np.float64)
    eps = float(assets["cf_cfg"].get("eps", 1e-7))

    # 4. inference
    dataset = LineTokensLabeledDataset(
        tokens_path, np.zeros(len(indices), dtype=np.int64),
        concentrations=conc_all[indices], indices=indices,
        aux_targets={k: v[indices] for k, v in aux_all.items()},
    )
    res = run_inference(module, dataset, args.batch_size, device, num_workers=args.num_workers)
    n_skipped = int(res["skipped"].sum())
    if n_skipped:
        print(f"WARNING: {n_skipped} spectra had no valid Voigt fit and were skipped (NaN rows)")

    # 5. outputs
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = run_dir / "evaluation" / f"cf_measured_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = conc_all[indices].astype(np.float64)
    sample_ids = sample_ids_all[indices]
    instruments = instruments_all[indices]

    rows = {
        "index": indices,
        "unique_id": unique_ids_all[indices],
        "sample_type_id": sample_ids,
        "instrument": instruments,
        "n_valid_lines": res["n_valid_lines"],
        "skipped": res["skipped"].astype(int),
        "cf_T": res["cf_T"], "cf_log10_Ne": res["cf_log10_Ne"],
        "T0": res["T0"], "log10_Ne0": res["log10_Ne0"], "log10_Nl0": res["log10_Nl0"],
        "n_lines_used": np.nansum(res["n_lines_used"], axis=1),
        "mean_weight": res["mean_weight"],
    }
    for i, e in enumerate(element_names):
        rows[f"pred_{e}"] = res["pred"][:, i]
        rows[f"true_{e}"] = y_true[:, i]
        rows[f"censored_{e}"] = res["censored"][:, i].astype(int)
        rows[f"binned_{e}"] = res["binned_pred"][:, i]
        rows[f"c0_{e}"] = res["c0"][:, i]
    pd.DataFrame(rows).to_csv(out_dir / "per_spectrum.csv", index=False)

    cf_table = per_element_table(element_names, y_true, res["pred"], lod, res["censored"],
                                 sample_ids, eps)
    have_binned = np.isfinite(res["binned_pred"]).any()
    binned_table = (per_element_table(element_names, y_true, res["binned_pred"], lod, None,
                                      sample_ids, eps) if have_binned else {})
    summary = {
        "cf": {lvl: macro_summary(cf_table, element_names, lvl)
               for lvl in ("spectrum_level", "sample_median")},
    }
    if have_binned:
        summary["binned_seed"] = {lvl: macro_summary(binned_table, element_names, lvl)
                                  for lvl in ("spectrum_level", "sample_median")}
    plasma = {
        "cf_T_median": float(np.nanmedian(res["cf_T"])),
        "cf_T_iqr": [float(np.nanpercentile(res["cf_T"], 25)), float(np.nanpercentile(res["cf_T"], 75))],
        "cf_log10_Ne_median": float(np.nanmedian(res["cf_log10_Ne"])),
        "cf_log10_Ne_iqr": [float(np.nanpercentile(res["cf_log10_Ne"], 25)),
                            float(np.nanpercentile(res["cf_log10_Ne"], 75))],
        "log10_Nl0_median": float(np.nanmedian(res["log10_Nl0"])),
        "mean_weight_median": float(np.nanmedian(res["mean_weight"])),
        "n_lines_used_median": float(np.nanmedian(np.nansum(res["n_lines_used"], axis=1))),
    }
    per_element_yaml = {
        "run_dir": str(run_dir),
        "libs_data_config": args.libs_data_config,
        "line_embedding_config": line_embedding_config,
        "tokens_path": str(tokens_path),
        "indices": args.indices,
        "split_strategy": strategy,
        "n_spectra": int(len(indices)),
        "n_skipped": n_skipped,
        "n_samples": int(len(np.unique(sample_ids))),
        "pure_physics": bool(args.pure_physics or (run_info.get("cf") or {}).get("pure_physics", False)),
        "summary": summary,
        "plasma": plasma,
        "cf": cf_table,
        "binned_seed": binned_table,
    }
    with open(out_dir / "per_element.yaml", "w") as f:
        yaml.dump(_py(per_element_yaml), f, sort_keys=False)

    per_inst: dict[str, dict] = {}
    for inst in sorted(np.unique(instruments)):
        sel = instruments == inst
        tbl = per_element_table(element_names, y_true[sel], res["pred"][sel], lod,
                                res["censored"][sel], sample_ids[sel], eps)
        per_inst[str(inst)] = {
            "n_spectra": int(sel.sum()),
            "n_samples": int(len(np.unique(sample_ids[sel]))),
            "summary": {lvl: macro_summary(tbl, element_names, lvl)
                        for lvl in ("spectrum_level", "sample_median")},
            "cf_T_median": float(np.nanmedian(res["cf_T"][sel])),
            "cf_log10_Ne_median": float(np.nanmedian(res["cf_log10_Ne"][sel])),
            "elements": {e: tbl[e] for e in element_names},
        }
    with open(out_dir / "per_instrument.yaml", "w") as f:
        yaml.dump(_py(per_inst), f, sort_keys=False)

    # 6. run_info.yaml: append test_results_measured (compact: summaries + per-element spectrum level)
    measured_results = {
        "timestamp": ts,
        "evaluation_dir": str(out_dir),
        "libs_data_config": args.libs_data_config,
        "spectra_cache_path": spectra_cache_path,
        "tokens_path": str(tokens_path),
        "indices": args.indices,
        "split_strategy": strategy,
        "n_spectra": int(len(indices)),
        "n_skipped": n_skipped,
        "pure_physics": per_element_yaml["pure_physics"],
        "summary": summary,
        "plasma": plasma,
        "per_element": {e: cf_table[e]["spectrum_level"] for e in element_names},
        "per_element_sample_median": {e: cf_table[e]["sample_median"] for e in element_names},
        "binned_seed_per_element": ({e: binned_table[e]["spectrum_level"] for e in element_names}
                                    if have_binned else None),
        "per_instrument": {k: {"n_spectra": v["n_spectra"], "summary": v["summary"]}
                           for k, v in per_inst.items()},
    }
    run_info_path = run_dir / "run_info.yaml"
    run_info_full = yaml.safe_load(open(run_info_path))
    run_info_full["test_results_measured"] = _py(measured_results)
    with open(run_info_path, "w") as f:
        yaml.dump(run_info_full, f, default_flow_style=False, sort_keys=False)

    # 7. console side-by-side
    print("\n" + "=" * 96)
    print(f"CF evaluation on {args.libs_data_config} — {len(indices)} spectra, "
          f"{per_element_yaml['n_samples']} samples → {out_dir}")
    print("=" * 96)
    print(f"Plasma: T median {plasma['cf_T_median']:.0f} K, log10 Ne median "
          f"{plasma['cf_log10_Ne_median']:.2f}, lines used median {plasma['n_lines_used_median']:.0f}")
    hdr = f"{'El':>3s} {'n_unc':>6s} | {'CF logRMSE':>10s} {'CF <2x':>7s} {'CF r2':>7s} {'cens':>5s}"
    if have_binned:
        hdr += f" | {'BIN logRMSE':>11s} {'BIN <2x':>7s} {'BIN r2':>7s}"
    hdr += f" | {'med logRMSE':>11s} {'med <2x':>7s}"
    print(hdr)
    for e in element_names:
        c = cf_table[e]["spectrum_level"]
        m = cf_table[e]["sample_median"]
        line = (f"{e:>3s} {int(c['n_uncensored_truth']):6d} | {c['log_rmse']:10.3f} "
                f"{c['within_2x']:7.3f} {c['r2']:7.3f} {int(c['n_censored']):5d}")
        if have_binned:
            b = binned_table[e]["spectrum_level"]
            line += f" | {b['log_rmse']:11.3f} {b['within_2x']:7.3f} {b['r2']:7.3f}"
        line += f" | {m['log_rmse']:11.3f} {m['within_2x']:7.3f}"
        print(line)
    for lvl in ("spectrum_level", "sample_median"):
        s = summary["cf"][lvl]
        line = (f"{lvl:>14s}: CF macro logRMSE {s['log_rmse_macro']:.3f}, within2x "
                f"{s['within_2x_macro']:.3f}, major r2 {s['r2_major']:.3f}")
        if have_binned:
            b = summary["binned_seed"][lvl]
            line += (f" | binned macro logRMSE {b['log_rmse_macro']:.3f}, within2x "
                     f"{b['within_2x_macro']:.3f}, major r2 {b['r2_major']:.3f}")
        print(line)
    print(f"\nWrote {out_dir / 'per_spectrum.csv'}, per_element.yaml, per_instrument.yaml; "
          f"appended test_results_measured to {run_info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot CF evaluation of a cf_quantification run on measured spectra")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="runs/finetune_<..> directory of a cf_quantification run")
    parser.add_argument("--libs_data_config", type=str, default="config/libs_data_measured.yaml",
                        help="libs data config to evaluate on (default: measured spectra)")
    parser.add_argument("--line_embedding_config", type=str, default=None,
                        help="line embedding config (default: the run's run_info value)")
    parser.add_argument("--pure_physics", action="store_true",
                        help="force the zero-parameter variant (classical weights, solver "
                             "default init) regardless of how the run was trained")
    parser.add_argument("--indices", type=str, choices=["all", "test"], default="all",
                        help="all spectra of the config, or its held-out test split")
    parser.add_argument("--split_strategy", type=str, choices=list(SPLIT_STRATEGIES), default=None,
                        help="split strategy for --indices test (default: config downstream.splits.strategy)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    main(parser.parse_args())
