"""Dataset-wide metrics, never averages of per-batch macro scores."""
import torch


def classification_metrics(target, predicted, num_classes):
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    predicted = torch.as_tensor(predicted, dtype=torch.long).cpu()
    if target.ndim != 1 or predicted.shape != target.shape or target.numel() == 0:
        raise ValueError('expected nonempty matching 1D labels/predictions')
    if ((target < 0) | (target >= num_classes) | (predicted < 0) |
            (predicted >= num_classes)).any():
        raise ValueError('class index out of range')
    cm = torch.bincount(target * num_classes + predicted,
                        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = cm.diag().double()
    support = cm.sum(1)
    denominator = support + cm.sum(0)
    f1 = torch.where(denominator > 0, 2 * tp / denominator, 0.)
    present = support > 0
    return {'accuracy': (tp.sum() / cm.sum()).item(),
            'macro_f1': f1.mean().item(),
            # Mean over classes with support; unsupported classes have no defined accuracy.
            'mean_per_class_accuracy': (tp[present] / support[present]).mean().item(),
            'per_class_accuracy': [(tp[i] / support[i]).item() if support[i] else None
                                   for i in range(num_classes)],
            'support': support.tolist(), 'confusion_matrix': cm.tolist()}


def reconstruction_error(prediction, target, states, pad_mask, vis=None):
    """Normalized mean Euclidean joint error; None when no MASKED joints exist.

    Pass `vis` to score only originally visible masked joints (the comparable metric for
    MSTD, whose masks also cover invisible joints with imputed, untrusted targets).
    """
    from .masking import JointState
    if prediction.shape != target.shape or target.shape != (*states.shape, 2):
        raise ValueError('reconstruction shape mismatch')
    if pad_mask.shape != states.shape[:2]:
        raise ValueError('padding shape mismatch')
    selected = (states == int(JointState.MASKED)) & pad_mask.unsqueeze(-1)
    if vis is not None:
        selected &= vis
    if not selected.any():
        return {'reconstruction_error': None, 'masked_joints': 0}
    errors = torch.linalg.vector_norm(prediction[selected] - target[selected], dim=-1)
    return {'reconstruction_error': errors.mean().item(), 'masked_joints': int(selected.sum())}
