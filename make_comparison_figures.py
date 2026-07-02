"""
Side-by-side publication figures comparing raw (intensity) vs tokenized
(line_token_linear) detection fine-tuning runs.

Usage:
    uv run python make_comparison_figures.py \\
        --run_dir_raw runs/finetune_2026-06-17_10-44-48_libs_detection_bin_4090 \\
        --run_dir_token runs/finetune_2026-06-14_20-14-42_libs_detection
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from make_publication_figures import (  # noqa: E402
    PUB_RC,
    FigureRegistry,
    _detection_per_element_metrics,
    _scalars,
    _select_detection_panels,
    draw_detection_panel,
    draw_embedding_map,
)
from publication.inference_runner import (  # noqa: E402
    FinetuneInferenceRunner,
    filter_indices_with_valid_tokens,
    load_pretrain_summary,
)

RAW_COLOR = "#0072B2"
TOKEN_COLOR = "#D55E00"

FAIR_RAW_PRETRAIN = "pretrain_2026-05-26_09-34-33_compare_raw_4090"
FAIR_TOKEN_PRETRAIN = "pretrain_2026-05-25_13-21-11_compare_token_linear_4090"


def validate_pairing(
    raw_info: dict,
    token_info: dict,
    strict_pretrain: bool = False,
) -> list[str]:
    """Return warning strings; raise on hard mismatches."""
    warnings: list[str] = []

    for key in ("task",):
        if raw_info.get(key) != token_info.get(key):
            raise ValueError(
                f"run pairing mismatch on {key}: "
                f"{raw_info.get(key)!r} vs {token_info.get(key)!r}",
            )
    if raw_info.get("task") != "detection":
        raise ValueError("comparison script currently supports task=detection only")

    if raw_info.get("element_names") != token_info.get("element_names"):
        raise ValueError("element_names differ between runs")

    for key in ("libs_data_config", "test_samples", "element_lod_config", "seed"):
        if raw_info.get(key) != token_info.get(key):
            raise ValueError(
                f"run pairing mismatch on {key}: "
                f"{raw_info.get(key)!r} vs {token_info.get(key)!r}",
            )

    raw_pt = load_pretrain_summary(raw_info.get("pretrain_run"))
    tok_pt = load_pretrain_summary(token_info.get("pretrain_run"))
    if raw_pt.get("path") != tok_pt.get("path"):
        warnings.append(
            "Pretrain runs differ — comparison reflects full pipelines, "
            "not an isolated embedding swap."
        )
    if raw_pt.get("pretrain_loss") != tok_pt.get("pretrain_loss"):
        warnings.append(
            f"Pretrain loss differs: raw={raw_pt.get('pretrain_loss')!r}, "
            f"token={tok_pt.get('pretrain_loss')!r}."
        )
    if raw_pt.get("epochs") != tok_pt.get("epochs"):
        warnings.append(
            f"Pretrain epoch count differs: raw={raw_pt.get('epochs')}, "
            f"token={tok_pt.get('epochs')}."
        )

    if strict_pretrain:
        for pt, fair in ((raw_pt, FAIR_RAW_PRETRAIN), (tok_pt, FAIR_TOKEN_PRETRAIN)):
            name = Path(str(pt.get("path", ""))).name
            if fair not in name:
                raise ValueError(
                    f"--strict_pretrain: expected {fair} in pretrain path, got {name}"
                )

    return warnings


def shared_test_indices(
    runner_raw: FinetuneInferenceRunner,
    runner_token: FinetuneInferenceRunner,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    test_idx = runner_raw.splits["test"]
    n_take = min(max_samples, len(test_idx))
    sub = np.sort(rng.choice(test_idx, size=n_take, replace=False))
    return filter_indices_with_valid_tokens(runner_token.tokens_path, sub)


def make_cmp_fig0_methodology(
    raw_info: dict,
    token_info: dict,
    warnings: list[str],
    reg: FigureRegistry,
):
    raw_pt = load_pretrain_summary(raw_info.get("pretrain_run"))
    tok_pt = load_pretrain_summary(token_info.get("pretrain_run"))
    det_raw = (raw_info.get("test_results") or {}).get("detection") or {}
    det_tok = (token_info.get("test_results") or {}).get("detection") or {}

    rows = [
        ["Embedding", raw_info.get("embedding_type"), token_info.get("embedding_type")],
        ["Pretrain run", Path(str(raw_pt.get("path", ""))).name,
         Path(str(tok_pt.get("path", ""))).name],
        ["Pretrain loss", raw_pt.get("pretrain_loss", "mse"),
         tok_pt.get("pretrain_loss", "mse")],
        ["Pretrain epochs", str(raw_pt.get("epochs", "?")),
         str(tok_pt.get("epochs", "?"))],
        ["Model params", str(raw_info.get("model_params")),
         str(token_info.get("model_params"))],
        ["Data config", raw_info.get("libs_data_config"),
         token_info.get("libs_data_config")],
        ["LOD config", raw_info.get("element_lod_config"),
         token_info.get("element_lod_config")],
        ["Test micro F1", f"{det_raw.get('micro_f1', 0):.3f}",
         f"{det_tok.get('micro_f1', 0):.3f}"],
        ["Test macro F1", f"{det_raw.get('macro_f1', 0):.3f}",
         f"{det_tok.get('macro_f1', 0):.3f}"],
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["", "Raw (intensity)", "Tokenized (line_token_linear)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    ax.set_title("Methodology: detection fine-tuning comparison", fontsize=13, pad=16)

    note = (
        "Note: results reflect the full pretrain + finetune pipeline. "
        "Pretrain objectives and schedules may differ between runs."
    )
    if warnings:
        note += " Warnings: " + " ".join(warnings)
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9.5, color="0.35",
             wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    reg.save(
        fig, "cmp_fig0_methodology",
        "Run metadata and test F1 for the raw vs tokenized detection models. "
        "Discloses pretrain differences that affect interpretability.",
    )


def make_cmp_fig1_aggregate(raw_info: dict, token_info: dict, reg: FigureRegistry):
    det_r = (raw_info.get("test_results") or {}).get("detection") or {}
    det_t = (token_info.get("test_results") or {}).get("detection") or {}
    metrics = ["micro_f1", "macro_f1", "micro_precision", "micro_recall"]
    labels = ["Micro F1", "Macro F1", "Micro P", "Micro R"]
    raw_vals = [float(det_r.get(m, 0)) for m in metrics]
    tok_vals = [float(det_t.get(m, 0)) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(x - w / 2, raw_vals, w, label="Raw", color=RAW_COLOR)
    ax.bar(x + w / 2, tok_vals, w, label="Tokenized", color=TOKEN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Aggregate detection metrics (test set)")
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig1_aggregate_metrics",
        "Grouped bars of micro/macro F1, precision, and recall on the held-out "
        "test split (from run_info.yaml).",
    )


def make_cmp_fig2_per_element(raw_info: dict, token_info: dict, reg: FigureRegistry):
    names = list(raw_info["element_names"])
    per_r = ((raw_info.get("test_results") or {}).get("detection") or {}).get(
        "per_element", {},
    )
    per_t = ((token_info.get("test_results") or {}).get("detection") or {}).get(
        "per_element", {},
    )
    raw_f1 = np.array([float(per_r.get(n, {}).get("f1", 0)) for n in names])
    tok_f1 = np.array([float(per_t.get(n, {}).get("f1", 0)) for n in names])
    order = np.argsort((raw_f1 + tok_f1) / 2)[::-1]
    ordered_names = [names[i] for i in order]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    ax.bar(x - w / 2, raw_f1[order], w, label="Raw", color=RAW_COLOR)
    ax.bar(x + w / 2, tok_f1[order], w, label="Tokenized", color=TOKEN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-element F1 (test)")
    ax.legend(loc="upper right")
    ax.set_title(f"Detection F1 across {len(names)} elements")
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig2_per_element_f1",
        "Per-element test F1 for raw and tokenized models, sorted by mean F1.",
    )


def make_cmp_fig3_delta(raw_info: dict, token_info: dict, reg: FigureRegistry):
    names = list(raw_info["element_names"])
    per_r = ((raw_info.get("test_results") or {}).get("detection") or {}).get(
        "per_element", {},
    )
    per_t = ((token_info.get("test_results") or {}).get("detection") or {}).get(
        "per_element", {},
    )
    delta = np.array([
        float(per_t.get(n, {}).get("f1", 0)) - float(per_r.get(n, {}).get("f1", 0))
        for n in names
    ])
    order = np.argsort(delta)
    colors = [TOKEN_COLOR if d >= 0 else RAW_COLOR for d in delta[order]]

    fig, ax = plt.subplots(figsize=(8.5, 11.5))
    ax.barh(range(len(names)), delta[order], color=colors, edgecolor="none")
    ax.axvline(0, color="0.3", lw=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("ΔF1 (tokenized − raw)")
    ax.set_title("Per-element F1 change")
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig3_delta_f1",
        "Per-element F1 difference (tokenized minus raw). Orange = token wins; "
        "blue = raw wins.",
    )


def make_cmp_fig4_training(
    raw_dir: Path,
    token_dir: Path,
    raw_info: dict,
    token_info: dict,
    reg: FigureRegistry,
):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for run_dir, label, color in (
        (raw_dir, "Raw", RAW_COLOR),
        (token_dir, "Tokenized", TOKEN_COLOR),
    ):
        va = _scalars(str(run_dir / "logs"), "val/det_f1")
        if va is not None:
            ax.plot(np.arange(1, len(va) + 1), va, lw=2, label=f"{label} val", color=color)
    test_r = (raw_info.get("test_results") or {}).get("test/det_f1")
    test_t = (token_info.get("test_results") or {}).get("test/det_f1")
    if test_r:
        ax.axhline(float(test_r), color=RAW_COLOR, ls=":", lw=1.2, alpha=0.7)
    if test_t:
        ax.axhline(float(test_t), color=TOKEN_COLOR, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detection F1")
    ax.set_title("Fine-tuning validation F1")
    ax.legend()
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig4_training_curves",
        "Validation detection F1 during fine-tuning for both models; dotted "
        "lines mark final test F1.",
    )


def make_cmp_fig5_embeddings(
    inf_raw: dict,
    inf_tok: dict,
    element_names: list[str],
    seed: int,
    reg: FigureRegistry,
):
    c_idx = element_names.index("C")
    color_raw = inf_raw["concentrations"][:, c_idx]
    color_tok = inf_tok["concentrations"][:, c_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    print("Computing t-SNE (raw)...")
    draw_embedding_map(ax1, fig, inf_raw["representations"], color_raw,
                       seed=seed, color_label="C content (wt.%)")
    ax1.set_title("Raw encoder embeddings", fontsize=12)
    print("Computing t-SNE (tokenized)...")
    draw_embedding_map(ax2, fig, inf_tok["representations"], color_tok,
                       seed=seed, color_label="C content (wt.%)")
    ax2.set_title("Tokenized encoder embeddings", fontsize=12)
    fig.suptitle("t-SNE on identical test subsample", fontsize=13, y=1.02)
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig5_embeddings_side_by_side",
        "t-SNE of pooled encoder embeddings on the same test spectra, colored "
        "by carbon concentration.",
    )


def make_cmp_fig6_panels(
    inf_raw: dict,
    inf_tok: dict,
    raw_info: dict,
    element_names: list[str],
    reg: FigureRegistry,
    n_panels: int = 6,
):
    per_elem = (
        (raw_info.get("test_results") or {}).get("detection") or {}
    ).get("per_element") or {}
    chosen = _select_detection_panels(per_elem, element_names, n=n_panels)
    if not chosen:
        chosen = list(range(min(n_panels, len(element_names))))

    fig, axes = plt.subplots(2, len(chosen), figsize=(3.1 * len(chosen), 6.8))
    if len(chosen) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, ei in enumerate(chosen):
        elem = element_names[ei]
        draw_detection_panel(
            axes[0, col], inf_raw["targets"][:, ei], inf_raw["probs"][:, ei],
            elem, color=RAW_COLOR, show_xlabel=False, show_ylabel=(col == 0),
        )
        draw_detection_panel(
            axes[1, col], inf_tok["targets"][:, ei], inf_tok["probs"][:, ei],
            elem, color=TOKEN_COLOR, show_xlabel=True, show_ylabel=(col == 0),
        )
        if col == 0:
            axes[0, col].set_ylabel("Raw\nP(present)")
            axes[1, col].set_ylabel("Tokenized\nP(present)")
    fig.suptitle("Presence detection panels (shared test subsample)", fontsize=13)
    fig.tight_layout()
    reg.save(
        fig, "cmp_fig6_detection_panels",
        f"Predicted presence probability vs LOD ground truth for the top "
        f"{len(chosen)} elements by test support; same spectra for both rows.",
    )


def make_cmp_fig7_composite(
    raw_info: dict,
    token_info: dict,
    inf_raw: dict,
    inf_tok: dict,
    element_names: list[str],
    reg: FigureRegistry,
):
    det_r = (raw_info.get("test_results") or {}).get("detection") or {}
    det_t = (token_info.get("test_results") or {}).get("detection") or {}
    names = list(element_names)
    per_r = det_r.get("per_element") or {}
    per_t = det_t.get("per_element") or {}
    delta = np.array([
        float(per_t.get(n, {}).get("f1", 0)) - float(per_r.get(n, {}).get("f1", 0))
        for n in names
    ])
    order = np.argsort(np.abs(delta))[::-1][:12]
    fe_idx = names.index("Fe") if "Fe" in names else int(order[0])

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.90, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, :])
    metrics = ["micro_f1", "macro_f1"]
    x = np.arange(2)
    w = 0.35
    ax_a.bar(x - w / 2, [det_r.get(m, 0) for m in metrics], w,
             label="Raw", color=RAW_COLOR)
    ax_a.bar(x + w / 2, [det_t.get(m, 0) for m in metrics], w,
             label="Tokenized", color=TOKEN_COLOR)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Micro F1", "Macro F1"])
    ax_a.set_ylim(0, 1.05)
    ax_a.legend()
    ax_a.set_title("Detection summary (test set)", fontsize=12)

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.barh(range(len(order)), delta[order],
              color=[TOKEN_COLOR if delta[i] >= 0 else RAW_COLOR for i in order])
    ax_b.axvline(0, color="0.3", lw=0.9)
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels([names[i] for i in order], fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("ΔF1")
    ax_b.set_title("Largest |ΔF1| elements", fontsize=11)

    ax_c = fig.add_subplot(gs[1, 1])
    draw_detection_panel(
        ax_c, inf_raw["targets"][:, fe_idx], inf_raw["probs"][:, fe_idx], "Fe",
        color=RAW_COLOR,
    )
    ax_c.set_title("Raw — Fe", fontsize=11)

    ax_d = fig.add_subplot(gs[1, 2])
    draw_detection_panel(
        ax_d, inf_tok["targets"][:, fe_idx], inf_tok["probs"][:, fe_idx], "Fe",
        color=TOKEN_COLOR,
    )
    ax_d.set_title("Tokenized — Fe", fontsize=11)

    fig.suptitle("LIBS foundation model: raw vs tokenized detection", fontsize=13)
    reg.save(
        fig, "cmp_fig7_summary_composite",
        "Graphical summary: aggregate F1, per-element delta F1, and Fe detection "
        "panels for both embedding modes.",
    )


def main(args: argparse.Namespace) -> None:
    plt.rcParams.update(PUB_RC)

    raw_dir = Path(args.run_dir_raw)
    token_dir = Path(args.run_dir_token)
    raw_info = yaml.safe_load(open(raw_dir / "run_info.yaml"))
    token_info = yaml.safe_load(open(token_dir / "run_info.yaml"))

    warnings = validate_pairing(raw_info, token_info, strict_pretrain=args.strict_pretrain)
    for w in warnings:
        print(f"WARNING: {w}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / "comparison_detection_raw_vs_token" / f"publication_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reg = FigureRegistry(output_dir)

    runner_raw = FinetuneInferenceRunner(
        raw_dir, device=args.device, batch_size=args.batch_size, label="raw",
    )
    runner_token = FinetuneInferenceRunner(
        token_dir, device=args.device, batch_size=args.batch_size, label="token",
    )

    shared_idx = shared_test_indices(
        runner_raw, runner_token, args.max_samples, args.seed,
    )
    print(f"Shared test subsample: {len(shared_idx)} spectra (seed={args.seed})")

    make_cmp_fig0_methodology(raw_info, token_info, warnings, reg)
    make_cmp_fig1_aggregate(raw_info, token_info, reg)
    make_cmp_fig2_per_element(raw_info, token_info, reg)
    make_cmp_fig3_delta(raw_info, token_info, reg)
    make_cmp_fig4_training(raw_dir, token_dir, raw_info, token_info, reg)

    inf_raw = runner_raw.run_detection_inference(shared_idx)
    inf_tok = runner_token.run_detection_inference(shared_idx)
    assert inf_raw["probs"].shape[0] == inf_tok["probs"].shape[0], (
        "inference row count mismatch between runs"
    )

    make_cmp_fig5_embeddings(
        inf_raw, inf_tok, raw_info["element_names"], args.seed, reg,
    )
    make_cmp_fig6_panels(
        inf_raw, inf_tok, raw_info, raw_info["element_names"], reg,
    )
    make_cmp_fig7_composite(
        raw_info, token_info, inf_raw, inf_tok, raw_info["element_names"], reg,
    )

    reg.write_readme([
        f"Raw run: {raw_info.get('run_name')}",
        f"Token run: {token_info.get('run_name')}",
        f"Task: detection",
        f"Shared test spectra: {len(shared_idx)}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Limitation: pretrain objectives/schedules may differ between runs. "
        "See cmp_fig0_methodology for details.",
    ])

    print("\n" + "=" * 60)
    print(f"Comparison figures complete: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw vs tokenized detection fine-tuning runs.",
    )
    parser.add_argument("--run_dir_raw", type=str, required=True)
    parser.add_argument("--run_dir_token", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict_pretrain",
        action="store_true",
        help="Abort unless pretrain runs match the fair 4090 comparison pair",
    )
    main(parser.parse_args())
