import torch
from torch import nn

class GaussianNoise(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.noiselog = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + torch.randn_like(x) * self.noiselog.exp()