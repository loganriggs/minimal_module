import torch
import torch.nn as nn

tau = 1
logits = torch.randn(6)
eps = 1e-10
uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)

noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
res = torch.sigmoid((logits + noise) / tau)
res = ((res > 0.5).type_as(res) - res).detach() + res

print(res)

model = nn.Sequential(
    nn.Linear(6, 400),
    nn.ReLU(),
    nn.Linear(400, 400),
    nn.ReLU(),
    nn.Linear(400, 2)
)



