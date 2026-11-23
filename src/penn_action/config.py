"""Small, strict YAML/dataclass config; dotted CLI overrides use YAML values."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml

INPUT_MODES = {'coords_only', 'coords_vis', 'coords_vis_mask', 'vis_only'}


@dataclass
class DataConfig:
    data_root: str = 'data/Penn_Action'
    val_ids: str = 'splits/val_ids.txt'
    schema: str = 'metadata/penn_action_schema.json'
    frames: int = 32
    normalization: str = 'visible_clip'
    hidden_policy: str = 'interpolate'
    debug_per_class: int = 0
    synthetic: bool = False


@dataclass
class ModelConfig:
    name: str = 'tcn'
    input_mode: str = 'coords_only'  # coords_only | coords_vis | coords_vis_mask | vis_only
    hidden_size: int = 64
    num_layers: int = 3  # tcn, transformer
    kernel_size: int = 5  # tcn only; odd
    dropout: float = 0.1  # tcn, transformer
    num_heads: int = 4  # transformer only; must divide hidden_size


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    scheduler: str = 'none'  # none | cosine (decays to zero over train.epochs)
    objective: str = 'classify'  # classify | pretrain (masked-joint reconstruction)
    mask_ratio: float = 0.3  # pretrain only
    mask_policy: str = 'visible'  # pretrain only: visible (MVA) | all (MSTD)
    init_checkpoint: str | None = None  # classify only: pretrained best.pt to fine-tune
    vis_dropout: float = 0.0  # classify only: hide up to this share of visible training joints
    workers: int = 0
    device: str = 'auto'  # auto: CUDA if available, else CPU
    output_dir: str = 'outputs/run'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        if self.model.input_mode not in INPUT_MODES:
            raise ValueError(f'input_mode must be one of {sorted(INPUT_MODES)}')
        if self.model.name == 'transformer' and (
                self.model.num_heads < 1 or self.model.hidden_size % self.model.num_heads):
            raise ValueError('num_heads must be positive and divide hidden_size')
        if self.train.objective not in {'classify', 'pretrain'}:
            raise ValueError('objective must be classify or pretrain')
        if self.train.objective == 'pretrain':
            if self.model.name != 'transformer' or self.model.input_mode != 'coords_vis_mask':
                raise ValueError('pretraining requires model.name=transformer and '
                                 'input_mode=coords_vis_mask')
            if not 0 < self.train.mask_ratio <= 1:
                raise ValueError('mask_ratio must be in (0, 1]')
            if self.train.mask_policy not in {'visible', 'all'}:
                raise ValueError('mask_policy must be visible or all')
            if self.train.init_checkpoint is not None:
                raise ValueError('init_checkpoint applies to fine-tuning, not pretraining')
            if self.train.vis_dropout:
                raise ValueError('vis_dropout applies to classification training, not pretraining')
        if not 0 <= self.train.vis_dropout < 1:
            raise ValueError('vis_dropout must be in [0, 1)')
        if self.train.vis_dropout and self.data.hidden_policy != 'interpolate':
            raise ValueError('vis_dropout requires hidden_policy=interpolate')
        if self.data.normalization not in {'visible_clip', 'none'}:
            raise ValueError('unknown normalization')
        if self.data.hidden_policy not in {'raw', 'interpolate'}:
            raise ValueError('hidden_policy must be raw or interpolate')
        if self.data.frames < 1 or self.data.debug_per_class < 0:
            raise ValueError('invalid frames/debug_per_class')
        if min(self.model.hidden_size, self.train.epochs, self.train.batch_size) < 1:
            raise ValueError('model size, epochs and batch size must be positive')
        if self.model.num_layers < 1 or self.model.kernel_size < 1 or self.model.kernel_size % 2 == 0:
            raise ValueError('num_layers must be positive and kernel_size a positive odd integer')
        if self.train.scheduler not in {'none', 'cosine'}:
            raise ValueError('scheduler must be none or cosine')
        if not 0 <= self.model.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if self.train.workers < 0 or self.train.learning_rate <= 0:
            raise ValueError('invalid workers/learning_rate')
        return self


def load_config(path, overrides=()):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict) or set(raw) - {'data', 'model', 'train'}:
        raise ValueError('config requires data/model/train sections only')
    for override in overrides:
        key, value = override.split('=', 1)
        section, name = key.split('.', 1)
        if section not in {'data', 'model', 'train'}:
            raise ValueError(f'unknown section: {section}')
        raw.setdefault(section, {})[name] = yaml.safe_load(value)
    cfg = Config(**{name: cls(**raw.get(name, {})) for name, cls in
                    [('data', DataConfig), ('model', ModelConfig), ('train', TrainConfig)]})
    # Reject YAML values with wrong scalar types (including bool in integer fields).
    defaults = asdict(Config())
    for section, values in asdict(cfg).items():
        for key, value in values.items():
            expected = defaults[section][key]
            if expected is None:
                if value is not None and not isinstance(value, str):
                    raise TypeError(f'{section}.{key} must be string or null')
            elif isinstance(expected, float):
                if type(value) not in (int, float):
                    raise TypeError(f'{section}.{key} must be numeric')
            elif type(value) is not type(expected):
                raise TypeError(f'{section}.{key} must be {type(expected).__name__}')
    return cfg.validate()
