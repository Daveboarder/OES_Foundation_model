# Project: LIBS Foundation Model

## Stage
Prototype

## Priority
Active

## Status
Pre-training and pretrain evaluation completed (local_fixed_masking run). Ready for fine-tuning.

## Goal
Build a self-supervised foundation model for LIBS spectroscopy using masked intensity prediction (MIP) on a Transformer encoder, then fine-tune for material classification and concentration regression.

## Last Session Summary
- **2026-03-18 (eval)**: Ran pretrain evaluation on `local_fixed_masking` checkpoint. Results: MSE=0.0066, RMSE=0.0814, MAE=0.0528, R²=0.8436. Plots and metrics saved to `runs/pretrain_2026-03-18_14-30-27_local_fixed_masking/evaluation/eval_2026-03-18_17-28-49/`.
- **2026-03-18**: Ran pre-training with fixed masking (`local_fixed_masking`). 5k samples, 10 epochs, bf16-mixed, ~4.8M params. Last checkpoint at epoch 8, val_loss=0.006635. Checkpoints saved in `runs/pretrain_2026-03-18_14-30-27_local_fixed_masking/checkpoints/`. Claude session crashed before state was updated — recovered in new session.
- **2026-03-11**: Major bug-fix session (see CHANGELOG.md). Fixed 4 critical bugs:
  1. BERT-style 80/10/10 masking was being nullified — embedding layer replaced ALL masked positions with mask token instead of only the 80% (type 1). Fixed in `training/pretrain.py`.
  2. `evaluate_model.py` pretrain evaluation completely broken — wrong dict keys, wrong model output, missing mask arg, wrong metric function signature. All fixed.
  3. Fine-tune checkpoint crash — `save_top_k=0` meant no checkpoint saved, test eval crashed on empty path. Changed to `save_top_k=1` in both `train_finetune.py` and `train_pretrain.py`.
  4. Wrong metric key names in finetune eval display (showed 0 instead of actual values).
- **2026-02-11**: W&B logging run recorded (`run-20260211_202319-mi1x7prn`).
- **Pre-2026-03-11**: Initial development — architecture, synthetic data, training pipeline, evaluation, run management all implemented and tested locally.

## Approaches Tried
- **Transformer encoder + MIP pretext task**: Core architecture implemented. ~3M params (d_model=256, 6 layers, 8 heads). ✅ Working
- **Contiguous + peak-biased masking**: Variable block sizes [25,50,100,150], 50% peak bias ratio. ✅ Implemented
- **BERT-style 80/10/10 token replacement**: 80% mask token, 10% random, 10% unchanged. ✅ Fixed (was broken before 2026-03-11)
- **Synthetic data (5 material classes)**: Fe, Cu, Al, Ca, Mixed — 2048-bin spectra. ✅ Working
- **Local pretrain run (`local_contiguous_test`)**: 10 epochs, best val_loss=0.0046 at epoch 9. ✅ Completed (pre-bug-fix, results may not reflect fixed masking)
- **Local pretrain run (`local_fixed_masking`)**: 10 epochs, 5k samples, ~4.8M params, bf16-mixed. val_loss=0.006635 at epoch 8. ✅ Completed (post-bug-fix, valid results)

## Results

### Pretrain Evaluation (local_fixed_masking, post-bug-fix) — 2026-03-18

| Metric | Value |
|--------|-------|
| MSE    | 0.0066 |
| RMSE   | 0.0814 |
| MAE    | 0.0528 |
| R²     | 0.8436 |

**Results directory:** `runs/pretrain_2026-03-18_14-30-27_local_fixed_masking/evaluation/eval_2026-03-18_17-28-49/`
- `pretrain_reconstruction_examples.png` — original vs predicted spectra with masked regions
- `pretrain_error_analysis.png` — error distribution, error by intensity, predicted vs true scatter
- `pretrain_embeddings_tsne.png` — t-SNE of CLS embeddings colored by material class
- `pretrain_metrics.txt` — numeric metrics

### Pretrain Evaluation (local_contiguous_test, pre-bug-fix) — DISCARDED

> Pre-bug-fix run with broken masking. Results were stored in `evaluation_results/` (now deleted). Metrics were unreliable due to the masking bug + different data generation code. Superseded by `local_fixed_masking`.

### Checkpoints Available
- `runs/pretrain_2026-03-18_14-30-27_local_fixed_masking/checkpoints/final_model.pt` — **current best** (post-bug-fix)
- `runs/pretrain_2026-03-18_14-30-27_local_fixed_masking/checkpoints/best.ckpt` — Lightning checkpoint
- `checkpoints/pretrain/local_contiguous_test/final_model.pt` — pre-bug-fix (legacy)
- `checkpoints/pretrain/local_contiguous_test/epoch=09-val_loss=0.0046.ckpt` — pre-bug-fix best (legacy)

## Current TODOs
- [x] Implement core architecture (Transformer encoder + MIP head)
- [x] Implement synthetic data generator (5 material classes)
- [x] Implement masking strategies (contiguous, peak-biased, BERT-style)
- [x] Implement pre-training pipeline (Lightning)
- [x] Implement fine-tuning pipeline (Lightning)
- [x] Implement evaluation with plots and metrics
- [x] Implement run management system
- [x] Fix BERT-style masking bug (was nullified in embedding layer)
- [x] Fix evaluate_model.py pretrain evaluation
- [x] Fix checkpoint saving (save_top_k=1)
- [x] Fix metric key names in finetune eval
- [x] **Run pre-training with fixed masking** — completed 2026-03-18 (`local_fixed_masking`)
- [ ] Run fine-tuning on new pretrained model
- [ ] Evaluate fine-tuned model (classification + regression)
- [ ] Replace synthetic data with simulations or real LIBS data @2026-03-27
- [ ] Explore higher mask ratios (30-40%) for sparse spectral data
- [ ] Consider curriculum masking, contrastive pre-training, multi-scale masking
- [ ] Send the embedding code to David @2026-03-26

## Blockers
None currently — codebase is functional, ready for training.

## Notes
- Reference paper in `LIBSSS/s41587-025-02663-3.pdf`
- Demo notebook at `notebooks/demo.ipynb`
- W&B integration available (logged run on 2026-02-11)
- TensorBoard logs in `logs/pretrain/`
- SLURM scripts available: `run_pretrain.slurm`, `run_finetune.slurm`, `run_eval.slurm`
- Three config tiers: local (5k samples, 16GB GPU), standard (50k, 24-40GB), A100 (500k, 500M params)
