import argparse, torch, torch.optim as optim, torch.nn as nn
from datasets.loaders import toy_loader
from models.stgcn import STGCN
from models.stdf import STDF
from models.astgcn import ASTGCN

def build_model(name, N, A):
    if name == 'stgcn':
        return STGCN(num_nodes=N, in_channels=1, out_channels=1, A=A)
    if name == 'stdf':
        return STDF(num_nodes=N, in_channels=1, out_channels=1, A=A, ext_dim=8)
    if name == 'astgcn':
        return ASTGCN(num_nodes=N, in_channels=1, out_channels=1, A=A)
    raise ValueError('unknown model')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['stgcn','stdf','astgcn'], default='stgcn')
    p.add_argument('--epochs', type=int, default=2)
    args = p.parse_args()

    x, y, A = toy_loader()
    model = build_model(args.model, x.size(2), A)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    for e in range(args.epochs):
        opt.zero_grad()
        if args.model == 'stdf':
            ext = torch.randn(x.size(0), 8, x.size(2), x.size(3))
            yhat = model(x, ext=ext)
        elif args.model == 'astgcn':
            static = torch.randn(x.size(0), 2, x.size(2))
            dynamic = torch.randn(x.size(0), 3, x.size(2), x.size(3))
            yhat = model(x, static=static, dynamic=dynamic)
        else:
            yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        print(f'Epoch {e+1}: loss={loss.item():.4f}')

if __name__ == '__main__':
    main()
