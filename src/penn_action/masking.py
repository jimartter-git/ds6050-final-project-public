"""Pretraining primitives; supervised training never samples masks."""
from enum import IntEnum
import torch

MASK_POLICIES = {'visible', 'all'}


class JointState(IntEnum):
    OBSERVED = 0
    NAT_MISSING = 1
    MASKED = 2


def joint_states(vis, pad_mask, ratio=0.0, generator=None, policy='visible'):
    """policy='visible' (MVA) may mask only visible real joints; 'all' (MSTD) any real slot."""
    if vis.dtype != torch.bool or pad_mask.dtype != torch.bool:
        raise TypeError('vis and pad_mask must be bool')
    if vis.ndim != 3 or pad_mask.shape != vis.shape[:2]:
        raise ValueError('expected vis [B,T,J] and pad_mask [B,T]')
    if not 0 <= ratio <= 1:
        raise ValueError('mask ratio must be in [0,1]')
    if policy not in MASK_POLICIES:
        raise ValueError(f'mask policy must be one of {sorted(MASK_POLICIES)}')
    valid = pad_mask.unsqueeze(-1).expand_as(vis)
    visible = vis & valid
    eligible = visible if policy == 'visible' else valid
    sampled = torch.rand(vis.shape, device=vis.device, generator=generator) < ratio
    masked = eligible & sampled
    states = torch.full_like(vis, int(JointState.NAT_MISSING), dtype=torch.long)
    states[visible] = int(JointState.OBSERVED)
    states[masked] = int(JointState.MASKED)
    return states, masked  # Padding is inert NAT_MISSING; always gate with pad_mask.


def refill_missing(coords, vis, masked, pad_mask):
    """Re-impute naturally missing joints from the observations that remain unmasked.

    Preprocessing interpolated each missing joint from every visible frame of the clip,
    including frames that pretraining then masks, so those values would carry the masked
    targets into the input. Here each missing slot is interpolated again along the input
    frames from unmasked observations of the same joint only: linear between neighbours,
    holding the nearest one at the ends, zero when none remains (as in data.fill_hidden).
    """
    if (coords.shape != (*vis.shape, 2) or masked.shape != vis.shape
            or pad_mask.shape != vis.shape[:2]):
        raise ValueError('coordinate/visibility/mask/padding shapes disagree')
    real = pad_mask.unsqueeze(-1).expand_as(vis)
    known = vis & ~masked & real
    frames = vis.shape[1]
    t = torch.arange(frames, device=vis.device).view(1, -1, 1).expand_as(vis)
    before = torch.where(known, t, torch.full_like(t, -1)).cummax(1).values
    after = torch.where(known, t, torch.full_like(t, frames)).flip(1).cummin(1).values.flip(1)
    lo = torch.where(before >= 0, before, after).clamp(0, frames - 1)
    hi = torch.where(after < frames, after, before).clamp(0, frames - 1)
    at = lambda index: coords.gather(1, index.unsqueeze(-1).expand(*index.shape, 2))
    weight = ((t - lo).clamp(min=0) / (hi - lo).clamp(min=1)).unsqueeze(-1).to(coords.dtype)
    filled = at(lo) + weight * (at(hi) - at(lo))
    filled = torch.where(((before >= 0) | (after < frames)).unsqueeze(-1), filled,
                         torch.zeros_like(filled))
    return torch.where((~vis & real).unsqueeze(-1), filled, coords)


def mask_coordinates(coords, states, pad_mask, fill=0.0):
    if coords.shape != (*states.shape, 2) or pad_mask.shape != states.shape[:2]:
        raise ValueError('coordinate/state/padding shapes disagree')
    # Only deliberate masks and padding are zeroed; pretraining refills natural gaps first.
    hidden = (states == int(JointState.MASKED)) | ~pad_mask.unsqueeze(-1)
    return torch.where(hidden.unsqueeze(-1), torch.full_like(coords, fill), coords)
