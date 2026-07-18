# models/tuning_modules/ssf.py
import torch
import torch.nn as nn

class SSF(nn.Module):
    def __init__(self, C, init_scale=1.0, init_shift=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(C) * float(init_scale))
        self.shift = nn.Parameter(torch.ones(C) * float(init_shift))

    def forward(self, x):
        s = self.scale.view(1, -1, 1, 1)
        b = self.shift.view(1, -1, 1, 1)
        return x * s + b
