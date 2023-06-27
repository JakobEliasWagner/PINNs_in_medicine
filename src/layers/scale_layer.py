"""
A scaling layer makes adding the physical constraints easier.
Training using scaled data works better for learning, but the inputs need to be scaled and unscaled to preserve their
physical properties.
Adding a scaled layer makes pytorch use the underlying code to ensure this property
"""

import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    def __init__(
        self,
        old_min: float,
        old_max: float,
        new_min: float = -1.0,
        new_max: float = 1.0,
    ):
        super().__init__()
        alpha = (new_max - new_min) / (old_max - old_min)
        beta = -alpha * old_min + new_min
        self.alpha = nn.Parameter(torch.FloatTensor(1).fill_(alpha))
        self.beta = nn.Parameter(torch.FloatTensor(1).fill_(beta))

    def forward(self, ipt):
        return ipt * self.alpha + self.beta
