import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A):
    A_hat = A + torch.eye(A.size(0), device=A.device)
    D_hat = torch.diag(torch.sum(A_hat, dim=1))
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels, bias=False)
        self.register_buffer('Abar', normalize_adj(A))

    def forward(self, x):
        # x: [B, T, N, C]
        x = torch.einsum('ij,btjc->btic', self.Abar, x)
        return self.theta(x)

class TemporalConv(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, 2*out_c, kernel_size=(1,k), padding=(0,k//2))
        self.out_c = out_c

    def forward(self, x):
        # x: [B, C, N, T]
        h = self.conv(x)
        P, Q = torch.split(h, self.out_c, dim=1)
        return P * torch.sigmoid(Q)

class STConvBlock(nn.Module):
    def __init__(self, in_c, out_c, A, k=3):
        super().__init__()
        self.t1 = TemporalConv(in_c, out_c, k)
        self.g = GraphConv(out_c, out_c, A)
        self.t2 = TemporalConv(out_c, out_c, k)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.t1(x)                # [B, C, N, T]
        x = x.permute(0,3,2,1)        # [B, T, N, C]
        x = self.g(x).permute(0,3,2,1)
        x = F.relu(x)
        x = self.t2(x)
        return self.bn(x)

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, A):
        super().__init__()
        self.block1 = STConvBlock(in_channels, 64, A)
        self.block2 = STConvBlock(64, 32, A)
        self.head = nn.Conv2d(32, out_channels, kernel_size=(1,1))

    def forward(self, x, **kwargs):
        # x: [B, C, N, T]
        x = self.block1(x)
        x = self.block2(x)
        y = self.head(x)              # [B, C', N, T]
        return y[..., -1:].contiguous()
