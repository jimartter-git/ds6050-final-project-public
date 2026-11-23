## How the shared code fits together

Dataset items have the same keys as collate with the batch axis removed; `clip_id` is a
string and `label` is a scalar int64 tensor. All item frames are real. Collate returns:

| Key | Shape/type | Meaning |
|---|---|---|
| `coords` | float32 `[B,T,J,2]` | normalized x/y; padding zero |
| `vis` | bool `[B,T,J]` | annotation visibility; padding false |
| `pad_mask` | bool `[B,T]` | True = real frame; trailing padding only |
| `label` | int64 `[B]` | index into saved alphabetically sorted classes |
| `clip_id` | list[str] | IDs for later per-clip analysis |

`J` comes from files, never a fixed 13 in the loader. `T` is the longest real sequence in
a batch after preprocessing, bounded by config.

Every model call in training and evaluation goes through `features.model_batch`, which
passes a model only the keys its input mode allows. Models never get the raw batch:

- `coords_only`: only `coords` and `pad_mask`; no visibility, states, labels, targets or
  clip IDs.
- `coords_vis`: also includes `vis` and `states` (not used by the reported models).
- `coords_vis_mask`: also includes `masked` (all false outside pretraining), giving the
  four-channel `(x,y,v,m)` input used by MT1, MSTD and MVA.
- `vis_only`: only `vis` and `pad_mask`; no coordinates (the V1 diagnostic).

`features.input_channels(mode)` gives the per-joint width (2, 3, 4, 1).

Models implement `forward(model_batch) -> {"logits": tensor[B,C]}` and optionally
`"recon": tensor[B,T,J,2]`. Add a model to `models/` and `models.build_model`, selected by
`model.name`; do not write a second preprocessing or training loop. The key filtering
prevents accidental leaks; it does not sandbox model code.

`masking.joint_states` builds `OBSERVED=0`, `NAT_MISSING=1`, `MASKED=2` from
visibility, padding, a masking ratio, and a policy: `visible` (MVA; only visible real
joint-frames can become `MASKED`) or `all` (MSTD; any non-padded slot).
`refill_missing` re-interpolates NAT_MISSING coordinates from the unmasked observations
only, then `mask_coordinates` zeroes MASKED coordinates and padding. The original batch
coordinates remain reconstruction targets. Padding uses an inert NAT_MISSING state and
always requires `pad_mask` gating.

`features.pretraining_batch(batch, ratio, generator, policy)` builds the pretraining
input: coords (natural gaps refilled from unmasked observations, masked slots zeroed),
pad_mask, vis (`v`), masked (`m`), and derived states. `v` means originally visible, even when deliberately masked; `m` marks deliberate
masks only. MSTD sets m=1 on its masked slots, like MVA; see experiments.md. MT1, MSTD and
MVA use the `coords_vis_mask` mode with m=0 during supervised training, fine-tuning and
eval, so their input widths match.

Pretraining (`train.objective: pretrain`, Transformer with `coords_vis_mask` only):
each batch is masked with `train.mask_ratio` and `train.mask_policy`; the loss is MSE
between `recon` and the original coordinates on all masked slots (including naturally
missing ones under MSTD, whose targets are imputed). Validation uses a fixed-seed mask
generator and selects the checkpoint with the lowest reconstruction error on masked
visible joints, which is comparable across policies. Fine-tuning sets
`train.init_checkpoint` to a pretrain `best.pt`; every weight except `classifier.*` is
loaded after the same dataset/model/preprocessing compatibility check as eval, and the
source checkpoint, seed, epoch and mask policy are recorded in `metadata.json`.

The Transformer (`models/transformer.py`) embeds each frame's flattened joints, adds a
learned position embedding (at most `data.frames` positions), runs a pre-norm encoder
with padding masked out, and returns `logits` from a masked mean over real frames and
`recon` `[B,T,J,2]` from every frame token.

Metrics use whole-split predictions: accuracy, macro-F1 over the complete class mapping
(zero F1 for unsupported classes), per-class accuracy (`null` for zero support), support,
and confusion matrix (rows=true, columns=predicted). Reconstruction is mean Euclidean
joint position error in normalized units on masked real joints only (optionally also
restricted to originally visible joints, as pretraining validation does). Empty masked
sets return `null`, not zero. Naturally invisible joints are never scored.


## Where things live

| Module | What it does |
|---|---|
| `data.py` | field validation, shared imputation/normalization, temporal selection, collate |
| `features.py` | which keys each model input mode may see; feature assembly |
| `masking.py` | joint states, deliberate masks, refilling natural gaps from unmasked observations |
| `models/` | the TCN and Transformer |
| `train.py` | shared training, validation selection, checkpoint and eval plumbing |
| `metrics.py` | whole-split metrics and masked-only reconstruction scoring |

Mean per-class accuracy averages per-class accuracy over classes with support.

Controlled occlusion (`--occlusion RATE --occlusion-seed S --occlusion-mode frames|spans`,
eval only): `data.occlude` sets v=0 on round(RATE × visible) joint-frames of each raw clip,
either single joint-frames or runs of 8–16 consecutive frames of one joint, then the
clip goes through the normal preprocessing, so hidden coordinates are re-imputed and
normalization uses only the remaining visible joints. It requires
`hidden_policy=interpolate`. The eval's `metadata.json` records `occlusion_rate`,
`occlusion_seed` and `occlusion_mode`. `train.vis_dropout` applies the same hiding at random
to training clips only (`data.drop_visibility`). `penn_action.analysis` holds the run
loading, McNemar and tertile code the scripts share.

Coordinate-only models never see labels, IDs, visibility or reconstruction targets.
Imputation and normalization do use visibility, the same way in every arm, so coordinates
can still hint at which joints were missing. Compare models only on the same preprocessing
and splits.
