# Experiment plan

What each model and comparison tests. Measured results are in [results.md](results.md).

## Models

Each frame has 13 joints. `v` is Penn Action's visibility flag; `m` marks joints we masked on
purpose, which only happens during pretraining (`m = 0` everywhere else).

| Model | Config | Input per joint | Role |
|---|---|---|---|
| MB0 | `mb0_tcn.yaml` | `x, y` | temporal CNN baseline |
| MT0 | `mt0_transformer.yaml` | `x, y` | coordinate-only Transformer |
| MT1 | `mt1_transformer_vis.yaml` | `x, y, v, m` | Transformer with the visibility flag |
| MT1-dropout | `mt1_vis_dropout.yaml` | `x, y, v, m` | MT1 trained with random visibility dropout |
| MSTD | `mstd_pretrain.yaml`, `mstd_finetune.yaml` | `x, y, v, m` | standard masked pretraining, then fine-tuning |
| MVA | `mva_pretrain.yaml`, `mva_finetune.yaml` | `x, y, v, m` | pretraining that masks only visible joints, then fine-tuning |
| V1 | `v1_vis_only.yaml` | `v` | visibility-only diagnostic |

The Transformers share one architecture: one token per frame, hidden size 64, 3 layers,
4 heads, dropout 0.1. MT1, MT1-dropout, MSTD and MVA use the same four-channel input, so
their sizes match. The TCN has 3 layers of 64 channels with kernel size 5.

## Research questions

| RQ | Comparison | Question |
|---|---|---|
| RQ1 | MT0 vs MT1 | Does a per-joint visibility flag improve classification? |
| RQ2a | MT1 vs MVA | Does masked pretraining that respects visibility beat supervised training alone? |
| RQ2b | MSTD vs MVA | Does masking and reconstructing only originally visible joints beat standard masking? |
| RQ3 | all models | Which models hold up when joints are hidden, naturally or on purpose? |

## Training

Supervised runs, and fine-tuning after pretraining: 50 epochs, Adam with learning rate
1e-3 and cosine decay to zero, batch size 32. The kept checkpoint is the epoch with the best
validation macro-F1. Every model is trained with seeds 42, 43 and 44.

MT1-dropout is MT1 trained while a random share, up to half, of each clip's visible joints
is hidden and re-interpolated every epoch (`train.vis_dropout: 0.5`). If that alone removes
MT1's sensitivity to hidden joints, pretraining is not what fixes it.

## Pretraining

Each step masks 30% of the eligible joint-frames; padding is never masked. MSTD can mask any
joint-frame, so some of its targets are interpolated values of invisible joints. MVA masks
only originally visible joints. Before masking, naturally missing joints are re-interpolated
from the observations that stay unmasked, so a masked value cannot reach the model through
its neighbours. Masked coordinates are then set to zero with `m = 1`.

A linear head reconstructs every joint's `x, y` from each frame token, with MSE on the
masked joint-frames. Pretraining uses the same optimizer, schedule and 50 epochs as above
and keeps the checkpoint with the lowest validation reconstruction error under fixed masks.
Fine-tuning starts from that checkpoint (all weights except the classifier) with the same
seed. This is a controlled version of CascadeFormer's random masking, not a reproduction.

For comparing MSTD and MVA, reconstruction error is measured only on masked joints that were
originally visible, since invisible joints have no true position. `scripts/summarize_training.py`
reports it at the kept checkpoint, along with training time, which gives the extra cost of
pretraining.

## Metrics and statistics

- Top-1 accuracy, macro-F1 and mean per-class accuracy (averaged over classes present), as
  mean ± sample standard deviation over the three seeds. Per-class accuracy and confusion
  matrices are saved with every evaluation.
- For RQ1, RQ2a and RQ2b, an exact two-sided McNemar test on the test predictions of each
  matched seed pair. Seeds are never pooled as if they were independent clips.
- The test split is only used for reported numbers, never to choose settings.

## Occlusion (RQ3)

- **Natural occlusion** (`scripts/occlusion_tertiles.py`): within each class, test clips are
  split into thirds by their share of invisible joint-frames among the 32 frames the model
  sees (ties broken by clip ID).
- **Controlled occlusion** (`scripts/occlusion_sweep.py`, then `scripts/occlusion_curve.py`):
  0–50% of each clip's visible joint-frames are hidden at test time (`v = 0`), and their
  coordinates are re-interpolated from what remains. Hiding happens on the full clip before
  frames are sampled. The hidden joints depend only on the occlusion seed and clip ID, so
  every model sees the same ones, and each higher rate hides a superset of the lower one.
  - `--occlusion-mode frames` hides single joint-frames. Interpolation recovers these almost
    exactly, so this mode is a weak test.
  - `--occlusion-mode spans` hides one joint at a time for 8–16 consecutive frames, about as
    long as joints stay hidden in the real labels (median 14 frames).

## Caveats

Visibility patterns can correlate with the action or the camera view. V1 shows how much the
visibility flag alone predicts, which is how a visibility shortcut shows up; it cannot by
itself explain why another model improves. Null or negative results are reported as they
are. Extra benchmarks, larger backbones, estimated poses and RGB input are out of scope.
