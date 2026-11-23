"""Load and check Penn Action labels, preprocess clips, and batch them with five keys."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


@dataclass
class Clip:
    clip_id: str
    coords: np.ndarray
    vis: np.ndarray
    action: str
    split_value: int
    bbox: np.ndarray | None = None
    dimensions: np.ndarray | None = None


def mat_fields(path):
    """Accept flat MATLAB variables or one struct; never assume a struct's name."""
    raw = {k: v for k, v in loadmat(path, simplify_cells=True).items()
           if not k.startswith('__')}
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))
    return raw


def read_clip(path, fields, dimensions_order=None):
    raw = mat_fields(path)
    required = {'x', 'y', 'visibility', 'action', 'split'}
    if not required <= fields.keys():
        raise ValueError(f'schema requires field roles {sorted(required)}')
    absent = [name for name in fields.values() if name not in raw]
    if absent:
        raise ValueError(f'{path}: missing mapped fields {absent}; available: {list(raw)}')
    # nframes is mandatory to disambiguate MATLAB singleton axes.
    if 'nframes' not in fields:
        raise ValueError('schema must map nframes')
    nframes = scalar_int(raw[fields['nframes']], 'nframes')
    if nframes < 1:
        raise ValueError(f'{path}: empty clip')
    arrays = [np.asarray(raw[fields[k]]) for k in ('x', 'y', 'visibility')]
    arrays = [a.reshape(1, -1) if nframes == 1 and a.ndim == 1 else a for a in arrays]
    if any(a.ndim != 2 or a.shape[0] != nframes for a in arrays):
        raise ValueError(f'{path}: expected time-first [nframes,J], got {[a.shape for a in arrays]}')
    x, y, vis = arrays
    if x.shape != y.shape or x.shape != vis.shape or x.shape[1] < 1:
        raise ValueError(f'{path}: x/y/visibility shapes disagree')
    if not np.isin(vis, [0, 1]).all():
        raise ValueError(f'{path}: visibility must be binary')
    coords = np.stack([x, y], axis=-1).astype(np.float32)
    # Fail explicitly even on invisible coordinates; never conceal invalid annotations.
    if not np.isfinite(coords).all():
        raise ValueError(f'{path}: nonfinite coordinates; inspect and choose a documented policy')
    action = str(raw[fields['action']]).strip()
    if not action or not isinstance(raw[fields['action']], str):
        raise ValueError(f'{path}: action must be a nonempty string')
    split_value = scalar_int(raw[fields['split']], 'split')
    bbox = None
    if 'bbox' in fields:
        bbox = np.asarray(raw[fields['bbox']], dtype=np.float32).reshape(nframes, -1)
        if bbox.shape != (nframes, 4) or not np.isfinite(bbox).all():
            raise ValueError(f'{path}: bbox must be finite [T,4]')
    dimensions = None
    if 'dimensions' in fields:
        if dimensions_order != 'hwt':
            raise ValueError('confirm dimensions_order=hwt before interpreting dimensions')
        dimensions = np.asarray(raw[fields['dimensions']]).reshape(-1)
        if dimensions.size != 3 or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
            raise ValueError(f'{path}: expected positive dimensions [H,W,T]')
        if dimensions[2] != nframes:
            raise ValueError(f'{path}: dimensions and nframes disagree')
    return Clip(Path(path).stem, coords, vis.astype(bool), action, split_value, bbox, dimensions)


def scalar_int(value, name):
    a = np.asarray(value)
    if a.size != 1 or not np.isfinite(a).all() or a.item() != int(a.item()):
        raise ValueError(f'{name} must be a scalar integer')
    return int(a.item())


def label_paths(data_root):
    paths = sorted((Path(data_root) / 'labels').glob('*.mat'))
    if not paths:
        raise ValueError(f'No labels/*.mat in {data_root}')
    return paths


def read_catalog(data_root, schema_path):
    schema = json.loads(Path(schema_path).read_text())
    clips = [read_clip(p, schema['fields'], schema.get('dimensions_order'))
             for p in label_paths(data_root)]
    classes = sorted({c.action for c in clips})
    joints = {c.coords.shape[1] for c in clips}
    if len(joints) != 1 or next(iter(joints)) != schema['joint_count']:
        raise ValueError('joint count differs across files or from inspected schema')
    if classes != schema['classes']:
        raise ValueError('class mapping differs from inspected schema; inspect again')
    expected = {schema['train_value'], schema['test_value']}
    if len(expected) != 2 or {c.split_value for c in clips} != expected:
        raise ValueError('split values differ from inspected schema')
    return clips, schema


def validation_split(clips, train_value, val_path, seed=42, fraction=0.2, create=False):
    """Save once. Never silently regenerate a missing or stale validation manifest."""
    path = Path(val_path)
    train = [c for c in clips if c.split_value == train_value]
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be between 0 and 1')
    if create:
        if path.exists():
            raise FileExistsError(f'{path} already exists; preserve team split')
        rng = np.random.default_rng(seed)
        val = []
        for action in sorted({c.action for c in train}):
            ids = sorted(c.clip_id for c in train if c.action == action)
            if len(ids) < 2:
                raise ValueError(f'{action}: need >=2 training clips for stratification')
            count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
            val.extend(rng.permutation(ids)[:count].tolist())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(sorted(val)) + '\n')
    if not path.exists():
        raise FileNotFoundError(f'Prepare and commit validation IDs first: {path}')
    lines = path.read_text().splitlines()
    val = set(lines)
    if len(val) != len(lines) or not val or not val <= {c.clip_id for c in train}:
        raise ValueError('validation IDs must be unique, nonempty and official-train only')
    for action in sorted({c.action for c in train}):
        ids = {c.clip_id for c in train if c.action == action}
        if not ids & val or not ids - val:
            raise ValueError(f'{action}: both train and val must contain this class')
    return val


def split_never_visible(clips):
    """Return (clips with any visible joint, IDs of clips with none)."""
    kept = [c for c in clips if c.vis.any()]
    return kept, sorted(c.clip_id for c in clips if not c.vis.any())


def fill_hidden(coords, vis):
    """Per-joint interpolation; endpoint hold. Entirely invisible joints use zero."""
    out = coords.copy()
    time = np.arange(len(coords))
    for j in range(coords.shape[1]):
        known = np.flatnonzero(vis[:, j])
        if len(known):
            for axis in range(2):
                out[:, j, axis] = np.interp(time, known, coords[known, j, axis])
        else:
            out[:, j] = 0
    return out


OCCLUSION_MODES = ('frames', 'spans')
SPAN_FRAMES = (8, 16)  # natural invisible stretches in Penn Action have a median of 14 frames


def occlude(vis, rate, seed, clip_id, mode='frames'):
    """Hide round(rate x visible) annotated joint-frames of one clip, at least one kept.

    mode='frames' hides single joint-frames at random. mode='spans' hides one joint at a time
    for 8-16 consecutive frames, which interpolation cannot recover. The order depends only
    on (seed, clip_id), so every model sees the same hidden joints and a higher rate hides a
    superset of a lower one.
    """
    if not 0 <= rate < 1:
        raise ValueError('occlusion rate must be in [0, 1)')
    if mode not in OCCLUSION_MODES:
        raise ValueError(f'occlusion mode must be one of {OCCLUSION_MODES}')
    flat = vis.ravel()
    visible = np.flatnonzero(flat)
    digest = hashlib.sha256(f'{seed}:{clip_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    count = min(int(np.floor(rate * len(visible) + 0.5)), max(len(visible) - 1, 0))
    if mode == 'frames':
        order = rng.permutation(visible)
    else:
        frames, joints = vis.shape
        order, seen = [], set()
        while len(order) < count:
            joint, start = rng.integers(joints), rng.integers(frames)
            length = rng.integers(SPAN_FRAMES[0], SPAN_FRAMES[1] + 1)
            for index in range(start * joints + joint, min(start + length, frames) * joints, joints):
                if flat[index] and index not in seen:
                    seen.add(index)
                    order.append(index)
        order = np.asarray(order, dtype=int)
    out = flat.copy()
    out[order[:count]] = False
    return out.reshape(vis.shape)


def drop_visibility(vis, max_rate):
    """Training augmentation: hide a random share, up to max_rate, of visible joint-frames.

    Draws from NumPy's global generator, which seeding and DataLoader workers already set.
    """
    visible = np.flatnonzero(vis.ravel())
    rate = np.random.uniform(0, max_rate)
    count = min(int(np.floor(rate * len(visible) + 0.5)), max(len(visible) - 1, 0))
    out = vis.ravel().copy()
    out[np.random.permutation(visible)[:count]] = False
    return out.reshape(vis.shape)


def preprocess(clip, frames, normalization, hidden_policy):
    coords, vis = clip.coords.copy(), clip.vis.copy()
    if hidden_policy == 'interpolate':
        coords = fill_hidden(coords, vis)
    elif hidden_policy != 'raw':
        raise ValueError('select raw or interpolate after label inspection')
    if normalization == 'visible_clip':
        if not vis.any():
            raise ValueError(f'{clip.clip_id}: cannot normalize a wholly invisible clip')
        visible = clip.coords[vis]
        center = visible.mean(0)
        scale = np.sqrt(np.mean(np.sum((visible - center) ** 2, axis=-1)))
        coords = (coords - center) / max(float(scale), 1e-6)
    elif normalization != 'none':
        raise ValueError('unknown normalization')
    if hidden_policy == "interpolate":
        # Zero in model coordinates, including after visible-clip normalization.
        coords[:, ~vis.any(axis=0)] = 0
    # Nearest-frame resampling preserves annotation visibility and coordinate correspondence.
    indices = np.round(np.linspace(0, len(coords) - 1, min(frames, len(coords)))).astype(int)
    return torch.from_numpy(coords[indices].astype(np.float32)), torch.from_numpy(vis[indices])


class PennActionDataset(Dataset):
    def __init__(self, clips, classes, frames=32, normalization='visible_clip', hidden_policy='raw',
                 occlusion=0.0, occlusion_seed=0, occlusion_mode='frames', vis_dropout=0.0):
        """`occlusion` hides that fraction of each clip's visible joint-frames (v=0) before the
        shared preprocessing, so hidden coordinates are re-imputed from what remains.
        `vis_dropout` does the same at random each time a clip is loaded (training only)."""
        if (occlusion or vis_dropout) and hidden_policy != 'interpolate':
            raise ValueError('controlled occlusion and vis_dropout require '
                             'hidden_policy=interpolate; raw would keep the hidden coordinates')
        self.clips = clips
        self.classes = classes
        self.class_to_index = {name: i for i, name in enumerate(classes)}
        self.frames, self.normalization, self.hidden_policy = frames, normalization, hidden_policy
        self.occlusion, self.occlusion_seed = occlusion, occlusion_seed
        self.occlusion_mode, self.vis_dropout = occlusion_mode, vis_dropout

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        if self.occlusion:
            clip = replace(clip, vis=occlude(clip.vis, self.occlusion, self.occlusion_seed,
                                             clip.clip_id, self.occlusion_mode))
        if self.vis_dropout:
            clip = replace(clip, vis=drop_visibility(clip.vis, self.vis_dropout))
        coords, vis = preprocess(clip, self.frames, self.normalization, self.hidden_policy)
        return {'coords': coords, 'vis': vis, 'pad_mask': torch.ones(len(coords), dtype=torch.bool),
                'label': torch.tensor(self.class_to_index[clip.action], dtype=torch.long),
                'clip_id': clip.clip_id}


def collate(batch):
    if not batch:
        raise ValueError('empty batch')
    joints = batch[0]['coords'].shape[1]
    length = max(len(item['coords']) for item in batch)
    coords = torch.zeros(len(batch), length, joints, 2, dtype=torch.float32)
    vis = torch.zeros(len(batch), length, joints, dtype=torch.bool)
    pad = torch.zeros(len(batch), length, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item['coords'])
        if n < 1 or item['coords'].shape != (n, joints, 2) or item['vis'].shape != (n, joints):
            raise ValueError('inconsistent item dimensions')
        if item['coords'].dtype != torch.float32 or item['vis'].dtype != torch.bool:
            raise TypeError('expected float32 coords / bool visibility')
        if item['pad_mask'].shape != (n,) or not item['pad_mask'].all():
            raise ValueError('dataset items must contain real frames only')
        coords[i, :n], vis[i, :n], pad[i, :n] = item['coords'], item['vis'], True
    return {'coords': coords, 'vis': vis, 'pad_mask': pad,
            'label': torch.stack([x['label'] for x in batch]),
            'clip_id': [x['clip_id'] for x in batch]}


def subsample(clips, per_class, seed):
    if per_class == 0:
        return clips
    rng = np.random.default_rng(seed)
    result = []
    for action in sorted({c.action for c in clips}):
        group = [c for c in clips if c.action == action]
        result.extend(group[i] for i in rng.permutation(len(group))[:per_class])
    return result


def synthetic_clips(seed=42):
    """Invented smoke fixture; these are not Penn Action classes/joint counts."""
    rng = np.random.default_rng(seed)
    return [Clip(f'synthetic-{i:03}', rng.normal(size=(5 + i % 5, 4, 2)).astype(np.float32),
                 rng.random((5 + i % 5, 4)) > .2, f'fixture-{i % 3}', 1 if i < 24 else -1)
            for i in range(36)]


def ids_hash(clips):
    return hashlib.sha256('\n'.join(sorted(c.clip_id for c in clips)).encode()).hexdigest()
