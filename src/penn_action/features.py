"""Decide which batch keys each model input mode may see, and build per-frame features."""
import torch
from .masking import joint_states, mask_coordinates, refill_missing

# Per-joint feature channels for each input mode.
INPUT_CHANNELS = {'coords_only': 2, 'coords_vis': 3, 'coords_vis_mask': 4, 'vis_only': 1}


def input_channels(input_mode):
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f'unknown input mode: {input_mode}')
    return INPUT_CHANNELS[input_mode]


def model_batch(batch, input_mode):
    # Labels, IDs, targets, visibility and state tensors cannot reach coords-only models.
    input_channels(input_mode)
    vis = batch['vis'] & batch['pad_mask'].unsqueeze(-1)
    if input_mode == 'vis_only':
        # Visibility-only diagnostic: no coordinates at all.
        return {'vis': vis, 'pad_mask': batch['pad_mask']}
    allowed = {'coords': batch['coords'], 'pad_mask': batch['pad_mask']}
    if input_mode in {'coords_vis', 'coords_vis_mask'}:
        allowed['vis'] = vis
        allowed['states'], _ = joint_states(batch['vis'], batch['pad_mask'])
    if input_mode == 'coords_vis_mask':
        # Supervised/fine-tune/eval input (x,y,v,m=0); pretraining sets m via pretraining_batch.
        allowed['masked'] = torch.zeros_like(vis)
    return allowed


def pretraining_batch(batch, ratio, generator=None, policy='visible'):
    """Masked-pretraining input (x,y,v,m); reconstruction targets remain in the raw batch.

    policy='visible' (MVA) masks only originally visible joints; policy='all' (MSTD) masks
    any non-padded joint slot. Naturally missing joints are refilled from the unmasked
    observations, so no masked target reaches the input; then masked coordinates are zeroed
    and m=1 is set on them.
    """
    allowed = model_batch(batch, 'coords_vis_mask')
    states, masked = joint_states(batch['vis'], batch['pad_mask'], ratio, generator, policy)
    refilled = refill_missing(batch['coords'], batch['vis'], masked, batch['pad_mask'])
    allowed['coords'] = mask_coordinates(refilled, states, batch['pad_mask'])
    allowed['states'], allowed['masked'] = states, masked
    return allowed


def frame_features(batch):
    """Concatenate per-joint channels in order (x, y, v, m) and flatten joints per frame."""
    parts = []
    if 'coords' in batch:
        parts.append(batch['coords'])
    if 'vis' in batch:
        parts.append(batch['vis'].unsqueeze(-1).float())
    if 'masked' in batch:
        parts.append(batch['masked'].unsqueeze(-1).float())
    return torch.cat(parts, dim=-1).flatten(2)
