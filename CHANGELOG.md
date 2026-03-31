# Changelog

All notable changes to the LIBS Foundation Model are documented here.

---

## [Unreleased] - 2026-03-11

### Fixed

#### Critical: BERT-style masking was being nullified (Bug #7)

The `SpectralEmbedding` layer was replacing **all** masked positions with the
learnable mask token, which completely overrode the BERT-style 80/10/10 token
replacement strategy applied by the dataset. The 10% "random value" and 10%
"unchanged" positions were silently being replaced with the mask token, meaning
the model only ever learned "when I see [MASK], predict something" instead of
learning general spectral context.

**Root cause:** `training/pretrain.py` passed `batch['mask']` (the full 15%
boolean mask) to the model's embedding layer. The embedding layer replaced
embeddings at *every* masked position with the mask token.

**Fix:** Now pass `embedding_mask = (batch['mask_type'] == 1)` — only the 80%
mask-token positions get the learnable mask embedding. The remaining 10% random
and 10% unchanged positions keep their intensity-projected embeddings. Loss is
still computed on all 15% masked positions.

**Files changed:** `training/pretrain.py` (training_step, validation_step)

**Impact:** The model now correctly implements BERT-style self-supervised learning:
- 80% of masked bins: see mask token embedding, must predict from context alone
- 10% of masked bins: see embedding of a random intensity value, must still predict correctly
- 10% of masked bins: see original embedding, acts as a regularizer
- Loss on all 15%, forcing the model to learn robust spectral representations

#### Critical: `evaluate_model.py` pretrain evaluation was completely broken

Multiple bugs prevented the pretrain evaluation from running at all:

1. **Wrong dictionary keys** (lines 206-208): Used `item['masked_spectrum']` and
   `item['original_spectrum']` but `MaskedLIBSDataset` returns `'input'` and
   `'target'`. Would crash with `KeyError`.

2. **Wrong model output** (line 211): Used `output['sequence_output']` (shape
   `[batch, 2049, 256]` — raw hidden states) instead of
   `output['mip_predictions']` (shape `[batch, 2048]` — actual predicted
   intensities). Would produce nonsensical results.

3. **Missing mask argument** (line 210): Called `model(masked_spectra)` without
   passing the mask, inconsistent with how the model was trained. Now passes
   `embedding_mask = (mask_types == 1)` to match training.

4. **Wrong `compute_mip_metrics` call** (lines 224-227): Called with 2 positional
   args (already-sliced arrays) but the function requires 3 args
   `(predictions, targets, mask)`. Would crash with `TypeError`.

**File changed:** `evaluate_model.py` (evaluate_pretrain function)

#### Critical: Fine-tuning test evaluation crashed on checkpoint load

`train_finetune.py` used `save_top_k=0` in `ModelCheckpoint`, which means
`best_model_path` returns an empty string. The post-training test step then
called `load_from_checkpoint("")` which crashes.

**Fix:** Changed to `save_top_k=1` so Lightning actually saves the best
checkpoint (by monitored metric). Same fix applied to `train_pretrain.py` for
consistency.

**Files changed:** `train_finetune.py`, `train_pretrain.py`

#### Minor: Wrong metric key names in finetune evaluation display

`evaluate_model.py` referenced `cls_metrics.get('macro_f1', 0)` and
`cls_metrics.get('weighted_f1', 0)` but `compute_classification_metrics`
returns keys `'f1_macro'` and `'f1_weighted'`. The display always showed 0.

**File changed:** `evaluate_model.py` (evaluate_finetune visualization)

### Summary of all files changed

| File | Changes |
|------|---------|
| `training/pretrain.py` | Pass `mask_type == 1` to embedding, full mask to loss |
| `evaluate_model.py` | Fix dict keys, model output, mask passing, metric call, metric keys |
| `train_finetune.py` | `save_top_k=0` → `save_top_k=1` |
| `train_pretrain.py` | `save_top_k=0` → `save_top_k=1` |
