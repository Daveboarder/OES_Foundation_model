#!/usr/bin/env bash
# End-to-end calibration-free pipeline on the physics_version-2 synthetic data.
#   1. regenerate synthetic spectra + CF dictionary + Voigt fits + tokens
#   2. CF tokens for the measured cache
#   3. re-pretrain the line_token_linear encoder on the CF tokens
#   4. seeds: quantification_binned + detection
#   5. cf_quantification (learned) and cf_quantification (pure physics)
#   6. zero-shot evaluation on measured spectra (+ classical CF with the 54 curated lines)
#   7. attention importance + publication figures for the learned run
# Usage: nohup bash scripts/run_cf_pipeline.sh > Outputs/cf_pipeline_<ts>/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
TS=$(date +%Y-%m-%d_%H-%M-%S)
OUT=Outputs/cf_pipeline_$TS
mkdir -p "$OUT"
STATE="$OUT/state.env"
touch "$STATE"
log() { echo "[$(date +%H:%M:%S)] $*"; }
run_dir_of() { grep -m1 "Run Directory:" "$1" | sed 's/.*Run Directory: *//' | tr -d '\r'; }
save() { echo "$1=$2" >> "$STATE"; }

DATA=config/libs_data.yaml
MEAS=config/libs_data_measured.yaml
LINE=config/line_embedding_cf.yaml
CFG=config/config_libs_token_linear_4090.yaml
CFCFG=config/config_libs_cf_4090.yaml

step() { log "=== STEP $1 ==="; }

step "1 tokens (synthetic regeneration + CF dictionary + Voigt fits)"
uv run python scripts/build_line_tokens.py --libs_data_config $DATA --line_embedding_config $LINE > "$OUT/01_tokens_synthetic.log" 2>&1 || { log "STEP 1 FAILED"; exit 1; }
grep -E "Tokens:|Shape:|Dictionary:|fit_valid" "$OUT/01_tokens_synthetic.log" | tail -5

step "2 tokens (measured)"
uv run python scripts/build_line_tokens.py --libs_data_config $MEAS --line_embedding_config $LINE > "$OUT/02_tokens_measured.log" 2>&1 || { log "STEP 2 FAILED"; exit 1; }
grep -E "Tokens:|Shape:" "$OUT/02_tokens_measured.log" | tail -2

step "3 pretrain"
uv run python train_pretrain.py --config $CFG --libs_data_config $DATA --line_embedding_config $LINE --experiment_name cf_v2_pretrain --num_workers 4 > "$OUT/03_pretrain.log" 2>&1 || { log "STEP 3 FAILED"; exit 1; }
PRE=$(run_dir_of "$OUT/03_pretrain.log"); save PRETRAIN "$PRE"; log "pretrain: $PRE"; grep -E "Best val loss" "$OUT/03_pretrain.log"

step "4a seed binned"
uv run python train_finetune.py --config $CFG --pretrain_run_dir "$PRE" --libs_data_config $DATA --line_embedding_config $LINE --task quantification_binned --pool cls_mean --experiment_name cf_v2_seed_binned --num_workers 4 > "$OUT/04a_seed_binned.log" 2>&1 || { log "STEP 4a FAILED"; exit 1; }
BIN=$(run_dir_of "$OUT/04a_seed_binned.log"); save SEED_BINNED "$BIN"; log "seed binned: $BIN"; grep -E "test/bin_accuracy|test/decoded_mae" "$OUT/04a_seed_binned.log" | head -3

step "4b seed detection"
uv run python train_finetune.py --config $CFG --pretrain_run_dir "$PRE" --libs_data_config $DATA --line_embedding_config $LINE --task detection --pool cls_mean --experiment_name cf_v2_seed_detection --num_workers 4 > "$OUT/04b_seed_detection.log" 2>&1 || { log "STEP 4b FAILED"; exit 1; }
DET=$(run_dir_of "$OUT/04b_seed_detection.log"); save SEED_DETECTION "$DET"; log "seed detection: $DET"; grep -E "test/det_f1" "$OUT/04b_seed_detection.log" | head -2

step "5a cf_quantification (learned weights)"
uv run python train_finetune.py --config $CFCFG --pretrain_run_dir "$PRE" --libs_data_config $DATA --line_embedding_config $LINE --task cf_quantification --pool cls_mean --seed_binned_run_dir "$BIN" --seed_detection_run_dir "$DET" --experiment_name cf_v2_cf_learned --num_workers 4 > "$OUT/05a_cf_learned.log" 2>&1 || { log "STEP 5a FAILED"; exit 1; }
CFL=$(run_dir_of "$OUT/05a_cf_learned.log"); save CF_LEARNED "$CFL"; log "cf learned: $CFL"; grep -E "test/cf_log_rmse|test/cf_within2x|test/te_mape|test/ne_log_mae" "$OUT/05a_cf_learned.log" | head -6

step "5b cf_quantification (pure physics)"
uv run python train_finetune.py --config $CFCFG --pretrain_run_dir "$PRE" --libs_data_config $DATA --line_embedding_config $LINE --task cf_quantification --pool cls_mean --seed_binned_run_dir "$BIN" --seed_detection_run_dir "$DET" --cf_pure_physics --experiment_name cf_v2_cf_pure --num_workers 4 > "$OUT/05b_cf_pure.log" 2>&1 || { log "STEP 5b FAILED"; exit 1; }
CFP=$(run_dir_of "$OUT/05b_cf_pure.log"); save CF_PURE "$CFP"; log "cf pure: $CFP"; grep -E "test/cf_log_rmse|test/cf_within2x" "$OUT/05b_cf_pure.log" | head -3

step "6 zero-shot evaluation on measured spectra"
uv run python scripts/evaluate_cf.py --run_dir "$CFL" --libs_data_config $MEAS --line_embedding_config $LINE --indices all > "$OUT/06a_eval_learned_measured.log" 2>&1 || log "STEP 6a FAILED (continuing)"
tail -3 "$OUT/06a_eval_learned_measured.log"
uv run python scripts/evaluate_cf.py --run_dir "$CFP" --libs_data_config $MEAS --line_embedding_config $LINE --indices all > "$OUT/06b_eval_pure_measured.log" 2>&1 || log "STEP 6b FAILED (continuing)"
tail -3 "$OUT/06b_eval_pure_measured.log"
MTOK=$(grep -m1 "Tokens:" "$OUT/02_tokens_measured.log" | awk '{print $2}')
MCACHE=$(ls -t external_data/cache/measured_cache_*.h5 | head -1)
LDICT=$(grep -m1 "Dictionary:" "$OUT/01_tokens_synthetic.log" | awk '{print $2}')
uv run python scripts/run_cf_classical.py --tokens "$MTOK" --spectra_cache "$MCACHE" --line_dict "$LDICT" --indices all --line_list cf_oes54 --out "$OUT/06c_classical_cf_oes54" > "$OUT/06c_classical_cf_oes54.log" 2>&1 || log "STEP 6c FAILED (continuing)"
tail -22 "$OUT/06c_classical_cf_oes54.log"
uv run python scripts/run_cf_classical.py --tokens "$MTOK" --spectra_cache "$MCACHE" --line_dict "$LDICT" --indices all --line_list all --out "$OUT/06d_classical_all" > "$OUT/06d_classical_all.log" 2>&1 || log "STEP 6d FAILED (continuing)"
tail -22 "$OUT/06d_classical_all.log"

step "7 attention importance + figures (learned run)"
uv run python analyze_attention_importance.py --run_dir "$CFL" --use_token_cache > "$OUT/07a_attention.log" 2>&1 || log "STEP 7a FAILED (continuing)"
uv run python make_publication_figures.py --run_dir "$CFL" > "$OUT/07b_figures.log" 2>&1 || log "STEP 7b FAILED (continuing)"
tail -5 "$OUT/07b_figures.log"

log "=== PIPELINE DONE ==="
cat "$STATE"
