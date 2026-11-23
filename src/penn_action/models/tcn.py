"""MB0: basic temporal CNN baseline over per-frame joint features."""
import torch
from torch import nn
from ..features import frame_features, input_channels


class TCNClassifier(nn.Module):
    def __init__(self, joints, classes, input_mode='coords_only', hidden_size=64,
                 num_layers=3, kernel_size=5, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd so each output aligns with its frame')
        sizes = [joints * input_channels(input_mode)] + [hidden_size] * num_layers
        self.convs = nn.ModuleList(nn.Conv1d(a, b, kernel_size, padding=kernel_size // 2)
                                   for a, b in zip(sizes, sizes[1:]))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, batch):
        features = frame_features(batch)
        real = batch['pad_mask'].unsqueeze(1).to(features.dtype)  # [B,1,T]
        lengths = real.sum(-1)
        if (lengths < 1).any():
            raise ValueError('each clip must have a real frame')
        x = features.transpose(1, 2) * real  # [B,features,T]
        for conv in self.convs:
            # Re-zero padding after every layer: bias/ReLU would otherwise leak into real frames.
            x = self.dropout(torch.relu(conv(x))) * real
        return {'logits': self.classifier(x.sum(-1) / lengths)}
