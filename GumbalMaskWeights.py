import torch
import torch.nn as nn
from torch import Tensor

class GumbalMaskWeights(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        # Add 1.75 to initialize to ~90% 1's for the mask
        self.logits_weights = nn.Parameter(torch.randn(out_features, in_features) + 1.75)
        self.logits_bias = nn.Parameter(torch.randn(out_features) + 1.75)
        self.eps = 1e-10
        self.tau = 1

    def forward(self, input: Tensor) -> Tensor:
        weight_mask = self.weight*self.make_mask(self.logits_weights)
        bias_mask = self.bias*self.make_mask(self.logits_bias)
        return nn.functional.linear(input, weight_mask, bias_mask)

    def make_mask(self, logits):
        uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
        noise = -((uniform[1] + self.eps).log() / (uniform[0] + self.eps).log() + self.eps).log()
        res = torch.sigmoid((logits + noise) / self.tau)
        mask = (res > 0.5).type_as(res) + res - res.detach()
        return mask