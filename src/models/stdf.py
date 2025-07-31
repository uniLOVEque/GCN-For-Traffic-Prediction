import torch
import torch.nn as nn
import torch.nn.functional as F
from .stgcn import STConvBlock, normalize_adj

class SETON(nn.Module):
    """Spatial Embedding by Temporal convolutiON.
    ext: dict of field->tensor (categorical one-hot or continuous), shape per node per time.
    This simplified version expects an external feature tensor Z: [B, F, N, T].
    """
    def __init__(self, in_f, k_space=16, k_time=16, t_kernel=3):
        super().__init__()
        self.space_embed = nn.Linear(in_f, k_space, bias=False)
        self.time_embed = nn.Conv1d(in_f, k_time, kernel_size=t_kernel, padding=t_kernel//2, bias=False)

    def forward(self, Z):
        # Z: [B, F, N, T]
        B, F, N, T = Z.shape
        s = self.space_embed(Z.permute(0,2,3,1)) # [B,N,T,k_space]
        t = self.time_embed(Z.view(B* N, F, T))  # [B*N,k_time,T]
        t = t.view(B, N, -1, T).permute(0,1,3,2) # [B,N,T,k_time]
        E = torch.cat([s, t], dim=-1).permute(0,3,1,2) # [B,k_space+k_time,N,T]
        return E

class STDF(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, A, ext_dim=8):
        super().__init__()
        self.stgcn1 = STConvBlock(in_channels, 32, A)
        self.seton = SETON(ext_dim, k_space=16, k_time=16)
        self.fuse = nn.Conv2d(32+32, 32, kernel_size=(1,1))
        self.stgcn2 = STConvBlock(32, 32, A)
        self.head = nn.Conv2d(32, out_channels, kernel_size=(1,1))

    def forward(self, x, ext=None, **kwargs):
        # x: [B, C, N, T]; ext: [B, F, N, T]
        h_g = self.stgcn1(x)
        if ext is None:
            ext = torch.zeros(x.size(0), 8, x.size(2), x.size(3), device=x.device)
        h_e = self.seton(ext)                 # [B, 32, N, T]
        h = torch.cat([h_g, h_e], dim=1)
        h = self.fuse(h)
        h = self.stgcn2(h)
        y = self.head(h)
        return y[..., -1:].contiguous()
