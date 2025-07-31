import torch
import numpy as np

def toy_loader(batch=4, N=20, T=12, C=1):
    x = torch.randn(batch, C, N, T)
    y = torch.randn(batch, C, N, 1)
    A = torch.rand(N, N)
    A = (A + A.T) / 2
    A.fill_diagonal_(1.0)
    return x, y, A
