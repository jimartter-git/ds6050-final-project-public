"""Shared model factory. Add models here; reuse the common training loop."""
from .tcn import TCNClassifier
from .transformer import TransformerClassifier


def build_model(config, joint_count, num_classes, max_frames=32):
    factories = {
        'tcn': lambda: TCNClassifier(joint_count, num_classes, config.input_mode,
                                     config.hidden_size, config.num_layers,
                                     config.kernel_size, config.dropout),
        'transformer': lambda: TransformerClassifier(joint_count, num_classes, config.input_mode,
                                                     config.hidden_size, config.num_layers,
                                                     config.num_heads, config.dropout,
                                                     max_frames),
    }
    if config.name not in factories:
        raise ValueError(f'unknown model: {config.name}; available: {sorted(factories)}')
    return factories[config.name]()


__all__ = ['TCNClassifier', 'TransformerClassifier', 'build_model']
