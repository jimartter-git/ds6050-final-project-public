"""MT0/MT1/MSTD/MVA: small Transformer encoder with per-frame tokens.

Each frame's joints x channels form one token; learned positional embeddings mark time.
Padded frames are excluded as attention keys and from pooling. The classification head
reads the mean over real frames; the reconstruction head predicts (x, y) for every joint
of every frame and is trained only during masked pretraining.
"""
import torch
from torch import nn
from ..features import frame_features, input_channels


class TransformerClassifier(nn.Module):
    def __init__(self, joints, classes, input_mode='coords_only', hidden_size=64,
                 num_layers=3, num_heads=4, dropout=0.1, max_frames=32):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        self.joints = joints
        self.embed = nn.Linear(joints * input_channels(input_mode), hidden_size)
        self.position = nn.Parameter(torch.zeros(1, max_frames, hidden_size))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 4 * hidden_size, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, classes)
        self.reconstruct = nn.Linear(hidden_size, joints * 2)

    def forward(self, batch):
        pad = batch['pad_mask']
        lengths = pad.sum(1, keepdim=True)
        if (lengths < 1).any():
            raise ValueError('each clip must have a real frame')
        frames = pad.shape[1]
        if frames > self.position.shape[1]:
            raise ValueError(f'{frames} frames exceed max_frames={self.position.shape[1]}')
        x = self.embed(frame_features(batch)) + self.position[:, :frames]
        x = self.norm(self.encoder(x, src_key_padding_mask=~pad))
        pooled = (x * pad.unsqueeze(-1).to(x.dtype)).sum(1) / lengths
        recon = self.reconstruct(x).view(*pad.shape, self.joints, 2)
        return {'logits': self.classifier(pooled), 'recon': recon}
