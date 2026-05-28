# RTX 4090 Raw vs Tokenized — Final Comparison Results

## Pretrain comparison (5 epochs, seed 42)

| Metric | Raw intensity (`config_libs_4090.yaml`, `embedding_type=intensity`) | Tokenized line-token linear (`config_libs_token_linear_4090.yaml`, `embedding_type=line_token_linear`) |
|---|---:|---:|
| Run directory | `runs/pretrain_2026-05-26_09-34-33_compare_raw_4090` | `runs/pretrain_2026-05-25_13-21-11_compare_token_linear_4090` |
| Seq length / tokens | ~17,428 wavelength bins + CLS | 2,425 lines + CLS |
| Model params | 819,201 | 812,674 |
| Train / Val samples | 95,965 / 16,935 | 95,965 / 16,935 |
| Cache used | N/A | `external_data/cache/line_tokens_ebea578d4de9.h5` |
| Best val loss | **0.000570** | 0.000683 |
| Last-epoch time (from log) | 42m 54s | 16m 31s |
| Wall time (start->finish, est.) | 3.61 h | 1.49 h |

## Finetune comparison (`task=quantification_binned`, `pool=cls_mean`, epochs=5, warmup=1)

| Metric | Raw (`compare_raw_ft_4090`) | Tokenized (`compare_token_linear_ft_4090`) |
|---|---:|---:|
| Run directory | `runs/finetune_2026-05-26_13-11-51_compare_raw_ft_4090` | `runs/finetune_2026-05-25_14-51-27_compare_token_linear_ft_4090` |
| Pretrain used | `runs/pretrain_2026-05-26_09-34-33_compare_raw_4090` | `runs/pretrain_2026-05-25_13-21-11_compare_token_linear_4090` |
| Best val/bin_accuracy | 0.8007 | **0.8380** |
| Test val/bin_accuracy | 0.8007 | **0.8377** |
| Test val/bin_loss (= val/loss) | 0.762 | **0.537** |
| Test decoded_mae (val/decoded_mae) | 0.00280 | **0.00156** |
| Test decoded_r2 (val/decoded_r2) | 0.155 | **-0.856** |
| Wall time (start->finish, est.) | 3.19 h | 1.70 h |

## Takeaway

Line-token linear (tokenized) **substantially improves binned quantification** on RTX 4090 finetuning: `val/bin_accuracy` is **~0.838 vs ~0.801** and `val/bin_loss` is much lower. It is also **significantly faster** (pretrain and finetune wall time), due to the shorter effective token sequence (2425 lines vs ~17428 bins).

Decoded regression quality (`decoded_r2`) behaves differently: raw intensity has positive `decoded_r2`, while tokenized has negative `decoded_r2`, suggesting bin classification accuracy does not automatically translate to decoded continuous spectra accuracy.

## Recommended next steps

1. Run longer finetuning budgets (e.g. epochs 10+) for both pipelines to reduce underfitting effects at 5 epochs, then re-check both `val/bin_accuracy` and decoded regression (`decoded_r2`).
2. Investigate the decoded head / decoding loss for the tokenized pipeline so that strong bin accuracy also improves decoded `R2` (e.g. adjust decoding calibration or add/weight decoded regression loss).
3. Keep using the token cache for throughput comparisons, and ensure masking/blocking settings remain identical across embedding modes when you do ablations.
