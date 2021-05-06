import torch
import torch.nn as nn

class GumbalMask(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        #Add 1.75 to initialize to ~90% 1's for the mask
        self.logits = nn.Parameter(torch.randn(size) + 1.75)
        self.eps = 1e-10
        self.tau = 1

    def forward(self, x):
        uniform = self.logits.new_empty([2] + list(self.logits.shape)).uniform_(0, 1)
        noise = -((uniform[1] + self.eps).log() / (uniform[0] + self.eps).log() + self.eps).log()
        res = torch.sigmoid((self.logits + noise) / self.tau)
        res = ((res > 0.5).type_as(res) - res).detach() + res
        return x*res


