# Traffic Prediction with Graph Convolutional Networks

A professionally curated comparison of **STGCN (2018)**, **STDF (2020)**, and **AST-GCN (2021)** for spatiotemporal traffic forecasting, plus PyTorch scaffolds and publication-ready figures.

- 📄 Blog: [`docs/posts/gcn-comparison.md`](docs/posts/gcn-comparison.md)
- 🖼 Figures: in `docs/assets/`
- 🧪 Code: in `src/`

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py --model stgcn --epochs 1
```

## Structure

```
traffic-prediction-gcn/
├── README.md
├── docs/
│   ├── index.md
│   ├── posts/
│   │   └── gcn-comparison.md
│   └── assets/
│       ├── stgcn.svg
│       ├── stdf.svg
│       └── astgcn.svg
├── src/
│   ├── datasets/
│   │   └── loaders.py
│   ├── models/
│   │   ├── stgcn.py
│   │   ├── stdf.py
│   │   └── astgcn.py
│   └── train.py
└── requirements.txt
```

## References

- STGCN: *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting* (AAAI 2018).
- STDF: *Spatiotemporal Data Fusion in Graph Convolutional Networks for Traffic Prediction* (IEEE Access 2020).
- AST-GCN: *Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting* (IEEE Access 2021).
