# Results

Measured results only. Values are mean ± sample standard deviation over seeds unless noted.
Test numbers come from one evaluation per seed of the best-validation-macro-F1 checkpoint.
The runs were made in our private development repo, so the Git commits cited below refer to
that history, not this one.

## MB0 — temporal CNN baseline, `(x, y)`

Run Nov 1, 2026 · `configs/mb0_tcn.yaml` · seeds 42, 43, 44 · CUDA · Git `8ef5c1b` plus
uncommitted MB0 changes · 50 epochs, Adam 1e-3, cosine decay, batch 32 · 1,004 train /
253 validation / 1,068 test clips.

### Test (1,068 clips)

| Metric | Mean ± std | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|---|
| Top-1 accuracy | 0.9023 ± 0.0081 | 0.9007 | 0.8951 | 0.9110 |
| Macro F1 | 0.8990 ± 0.0088 | 0.8970 | 0.8913 | 0.9086 |
| Mean per-class accuracy | 0.9015 ± 0.0072 | 0.8995 | 0.8955 | 0.9095 |

### Validation at the selected checkpoint (253 clips)

| Seed | Best epoch | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|---|
| 42 | 33 | 0.9170 | 0.9180 | 0.9162 |
| 43 | 28 | 0.9130 | 0.9119 | 0.9129 |
| 44 | 28 | 0.9209 | 0.9254 | 0.9194 |

### Test per-class accuracy

| Class | Test clips | Mean ± std |
|---|---|---|
| baseball_pitch | 63 | 0.9365 ± 0.0275 |
| baseball_swing | 58 | 0.9310 ± 0.0299 |
| bench_press | 71 | 0.8685 ± 0.0293 |
| bowl | 85 | 0.8980 ± 0.0245 |
| clean_and_jerk | 45 | 0.8444 ± 0.0222 |
| golf_swing | 77 | 0.9048 ± 0.0075 |
| jump_rope | 42 | 0.9444 ± 0.0137 |
| jumping_jacks | 56 | 0.9107 ± 0.0179 |
| pullup | 101 | 0.9439 ± 0.0114 |
| pushup | 107 | 0.9595 ± 0.0195 |
| situp | 50 | 0.8600 ± 0.0200 |
| squat | 117 | 0.8803 ± 0.0342 |
| strum_guitar | 48 | 0.9444 ± 0.0241 |
| tennis_forehand | 77 | 0.7662 ± 0.0390 |
| tennis_serve | 71 | 0.9296 ± 0.0373 |

### Test by natural-occlusion tertile (within class)

Same test evaluations; `scripts/occlusion_tertiles.py`. Occlusion rate = share of
annotated-invisible joint slots in the 32 frames the model sees.

| Tertile | Clips | Rate range (mean) | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|---|---|
| Low | 360 | 0.000–0.385 (0.108) | 0.9426 ± 0.0070 | 0.9400 ± 0.0079 | 0.9429 ± 0.0076 |
| Medium | 358 | 0.000–0.469 (0.206) | 0.9376 ± 0.0032 | 0.9384 ± 0.0045 | 0.9472 ± 0.0028 |
| High | 350 | 0.000–0.805 (0.357) | 0.8248 ± 0.0144 | 0.8132 ± 0.0145 | 0.8123 ± 0.0117 |

### Test under controlled occlusion

Same checkpoints; `scripts/occlusion_sweep.py` with occlusion seed 0, then
`scripts/occlusion_curve.py`. Each rate hides that share of each clip's visible
joint-frames (v=0, coordinates re-imputed by the shared preprocessing).

| Hidden | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|
| 0% | 0.9023 ± 0.0081 | 0.8990 ± 0.0088 | 0.9015 ± 0.0072 |
| 10% | 0.8995 ± 0.0099 | 0.8958 ± 0.0111 | 0.8983 ± 0.0092 |
| 20% | 0.8983 ± 0.0113 | 0.8945 ± 0.0122 | 0.8975 ± 0.0108 |
| 30% | 0.8983 ± 0.0095 | 0.8950 ± 0.0103 | 0.8979 ± 0.0091 |
| 40% | 0.8948 ± 0.0090 | 0.8913 ± 0.0102 | 0.8939 ± 0.0078 |
| 50% | 0.8914 ± 0.0092 | 0.8880 ± 0.0105 | 0.8907 ± 0.0085 |

## MT0, MT1, V1 — supervised Transformers

Run Nov 6, 2026 · `configs/mt0_transformer.yaml` `(x, y)`, `configs/mt1_transformer_vis.yaml`
`(x, y, v, m)`, `configs/v1_vis_only.yaml` `(v)` · seeds 42, 43, 44 · UVA Rivanna, NVIDIA
RTX A6000 · Git `3ce9625`, clean checkout, configs unchanged (same budget and splits as
MB0) · commands as in the README, run as Slurm batch jobs. MB0 was rerun in the same jobs
for a matched comparison (last subsection).

### Validation at the selected checkpoint (253 clips)

The seed-42 rows are the one-seed MB0/MT0/MT1 comparison requested for Milestone II.

| Model | Seed | Best epoch | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|---|---|
| MB0 (rerun) | 42 | 24 | 0.9170 | 0.9185 | 0.9159 |
| MT0 | 42 | 15 | 0.9565 | 0.9564 | 0.9557 |
| MT1 | 42 | 19 | 0.9565 | 0.9577 | 0.9527 |
| MT0 | 43 | 16 | 0.9526 | 0.9470 | 0.9500 |
| MT1 | 43 | 12 | 0.9447 | 0.9467 | 0.9465 |
| MT0 | 44 | 17 | 0.9644 | 0.9651 | 0.9611 |
| MT1 | 44 | 19 | 0.9605 | 0.9590 | 0.9556 |
| V1 | 42 / 43 / 44 | 41 / 32 / 43 | 0.6443 / 0.6680 / 0.6482 | 0.6059 / 0.6498 / 0.6245 | 0.6178 / 0.6549 / 0.6290 |

### Test (1,068 clips)

| Model | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|
| MB0 (rerun) | 0.9020 ± 0.0052 | 0.8982 ± 0.0060 | 0.9008 ± 0.0044 |
| MT0 | 0.9329 ± 0.0084 | 0.9295 ± 0.0083 | 0.9316 ± 0.0080 |
| MT1 | 0.9366 ± 0.0084 | 0.9333 ± 0.0087 | 0.9341 ± 0.0080 |
| V1 | 0.6526 ± 0.0043 | 0.6394 ± 0.0068 | 0.6429 ± 0.0066 |

### RQ1: MT0 vs MT1, McNemar's test per matched seed (test)

`scripts/mcnemar.py`. No seed shows a significant difference.

| Seed | MT0 accuracy | MT1 accuracy | Only MT0 correct | Only MT1 correct | p |
|---|---|---|---|---|---|
| 42 | 0.9382 | 0.9419 | 19 | 23 | 0.644 |
| 43 | 0.9232 | 0.9270 | 29 | 33 | 0.704 |
| 44 | 0.9373 | 0.9410 | 19 | 23 | 0.644 |

### Test by natural-occlusion tertile (within class)

Same tertiles as MB0 (360 / 358 / 350 clips).

| Tertile | Model | Accuracy | Macro F1 | Mean per-class accuracy |
|---|---|---|---|---|
| Low | MT0 | 0.9611 ± 0.0028 | 0.9585 ± 0.0028 | 0.9611 ± 0.0022 |
| Low | MT1 | 0.9657 ± 0.0042 | 0.9649 ± 0.0043 | 0.9653 ± 0.0046 |
| Low | V1 | 0.5602 ± 0.0163 | 0.5592 ± 0.0173 | 0.5682 ± 0.0184 |
| Medium | MT0 | 0.9516 ± 0.0138 | 0.9512 ± 0.0132 | 0.9582 ± 0.0120 |
| Medium | MT1 | 0.9665 ± 0.0028 | 0.9658 ± 0.0033 | 0.9690 ± 0.0030 |
| Medium | V1 | 0.7514 ± 0.0201 | 0.7453 ± 0.0225 | 0.7478 ± 0.0220 |
| High | MT0 | 0.8848 ± 0.0100 | 0.8753 ± 0.0100 | 0.8742 ± 0.0104 |
| High | MT1 | 0.8762 ± 0.0194 | 0.8651 ± 0.0209 | 0.8663 ± 0.0178 |
| High | V1 | 0.6467 ± 0.0147 | 0.6112 ± 0.0114 | 0.6126 ± 0.0137 |

### Test under controlled occlusion

Same procedure and occlusion seed (0) as MB0. Accuracy:

| Hidden | MT0 | MT1 | V1 |
|---|---|---|---|
| 0% | 0.9329 ± 0.0084 | 0.9366 ± 0.0084 | 0.6526 ± 0.0043 |
| 10% | 0.9304 ± 0.0064 | 0.9338 ± 0.0066 | 0.3099 ± 0.0150 |
| 20% | 0.9295 ± 0.0070 | 0.9273 ± 0.0035 | 0.2194 ± 0.0174 |
| 30% | 0.9282 ± 0.0081 | 0.9195 ± 0.0089 | 0.1782 ± 0.0070 |
| 40% | 0.9270 ± 0.0068 | 0.9023 ± 0.0135 | 0.1651 ± 0.0163 |
| 50% | 0.9248 ± 0.0085 | 0.8811 ± 0.0147 | 0.1511 ± 0.0159 |

Macro F1:

| Hidden | MT0 | MT1 | V1 |
|---|---|---|---|
| 0% | 0.9295 ± 0.0083 | 0.9333 ± 0.0087 | 0.6394 ± 0.0068 |
| 10% | 0.9264 ± 0.0069 | 0.9308 ± 0.0062 | 0.2795 ± 0.0186 |
| 20% | 0.9255 ± 0.0073 | 0.9236 ± 0.0029 | 0.1761 ± 0.0146 |
| 30% | 0.9246 ± 0.0087 | 0.9145 ± 0.0110 | 0.1231 ± 0.0080 |
| 40% | 0.9233 ± 0.0072 | 0.8964 ± 0.0178 | 0.1078 ± 0.0212 |
| 50% | 0.9208 ± 0.0089 | 0.8744 ± 0.0217 | 0.0932 ± 0.0219 |

### MB0 rerun at `3ce9625`

Same jobs and hardware as the Transformers. Test accuracy 0.9020 ± 0.0052 against 0.9023 ±
0.0081 recorded above; best epochs 24 / 28 / 37 against 33 / 28 / 28 (the Nov 1 run had
uncommitted changes). Seed 42 matches `outputs/mb0-my-seed42` (epoch 24, validation
accuracy 0.9170, macro F1 0.9185). Tertile and controlled-occlusion results for the rerun:

| | Low | Medium | High | 0% hidden | 50% hidden |
|---|---|---|---|---|---|
| Accuracy | 0.9444 ± 0.0056 | 0.9330 ± 0.0048 | 0.8267 ± 0.0087 | 0.9020 ± 0.0052 | 0.8933 ± 0.0057 |

MSTD and MVA were also run end to end as a pipeline check, before pretraining refilled
naturally missing joints from unmasked observations only. A masked target could then reach
the input through neighbouring frames, so those numbers are not reported.

`scripts/plot_results.py` draws the seed-42 training curves and the controlled-occlusion
figure from these runs.
