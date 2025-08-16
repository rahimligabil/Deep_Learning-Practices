import torch.nn as nn

def build_linear(input_dim: int):
    return nn.Sequential(nn.Linear(input_dim, 1, bias=True))
