import torch
import torch.nn as nn

def gumbal(logits):
    tau = 1
    eps = 1e-10
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)
    res = ((res > 0.5).type_as(res) - res).detach() + res
    # print(res)
    return res

def average(logits):
    average_runs = 10000
    sum = 0
    for x in range(average_runs):
        sum += torch.count_nonzero(gumbal(logits))
    return sum/average_runs

ave = average(torch.rand(100) + 1.725)