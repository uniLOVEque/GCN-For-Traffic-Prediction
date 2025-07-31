import torch
import torch.nn as nn
import torch.nn.functional as F
from .stgcn import GraphConv, TemporalConv, normalize_adj

class AttributeAugment(nn.Module):
    """Concatenate static S and windowed dynamic D into the feature channels.
    static: [B, P, N]  -> expand to [B, P, N, T]
    dynamic: [B, W, N, T] (already aligned)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, static=None, dynamic=None):
        # x: [B, C, N, T]
        feats = [x]
        if static is not None:
            B, P, N = static.shape
            T = x.size(-1)
            S = static.unsqueeze(-1).expand(B, P, N, T)
            feats.append(S)
        if dynamic is not None:
            feats.append(dynamic)  # assume [B, W, N, T]
        return torch.cat(feats, dim=1)

class ASTGCNCell(nn.Module):
    def __init__(self, in_c, hid_c, A):
        super().__init__()
        self.gcn = GraphConv(in_c, hid_c, A)
        self.gru = nn.GRU(input_size=hid_c, hidden_size=hid_c, batch_first=True)

    def forward(self, x):
        # x: [B, C, N, T]
        x = x.permute(0,3,2,1)           # [B,T,N,C]
        g = self.gcn(x)                  # [B,T,N,H]
        B,T,N,H = g.shape
        g = g.view(B*T, N, H)
        # pool over neighbors before GRU (simplified)
        g = g.mean(dim=1)                # [B*T, H]
        g = g.view(B, T, H)
        out, _ = self.gru(g)             # [B,T,H]
        return out[:, -1, :]

class ASTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, A, hid=64):
        super().__init__()
        self.aug = AttributeAugment()
        self.cell = ASTGCNCell(in_channels, hid, A)
        self.fc = nn.Linear(hid, out_channels * num_nodes)

    def forward(self, x, static=None, dynamic=None, **kwargs):
        x_aug = self.aug(x, static, dynamic)   # [B, C', N, T]
        h = self.cell(x_aug)                   # [B, H]
        y = self.fc(h).view(x.size(0), -1, x.size(2), 1)
        return y
