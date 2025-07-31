
# 基于图卷积网络的交通预测方法技术解析

> 本文对比三种代表性方法：**STGCN (2018)**、**STDF (2020)**、**AST-GCN (2021)**，从建模假设、结构设计、外部数据融合、复杂度与工程可落地性等维度展开系统解析，并给出可复现的代码骨架与图示。

---

## 1. 问题定义与符号

- 路网以图 \( G=(V, E, A) \) 表示，\( |V|=N \) 为节点数，\(A\in\mathbb{R}^{N\times N}\) 为加权邻接矩阵（可包含自环）。
- 历史观测构成张量 \(X\in \mathbb{R}^{B\times C\times N\times T}\)：批大小 \(B\)，通道数 \(C\)（如速度/流量），节点数 \(N\)，时间窗口长度 \(T\)。
- 目标是在给定 \(X\) 与可选外部因子 \(Z\) 的情况下预测未来 \(H\) 个时间步：\(\hat{Y}\in\mathbb{R}^{B\times C'\times N\times H}\)。

### 图卷积（谱域近似）
谱域图卷积可由切比雪夫多项式近似：
\[
\Theta *_{\mathcal{G}} X \approx \sum_{k=0}^{K-1}\theta_k T_k(\tilde{L}) X, \quad \tilde{L}=\frac{2}{\lambda_{\max}}L-I.
\]
一阶近似可化为常用的归一化形式：
\[
\hat{A}=A+I,\ \hat{D}_{ii}=\sum_j \hat{A}_{ij},\ \bar{A}=\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}.
\]

---

## 2. STGCN（2018）：全卷积式时空图卷积框架

**要点**：通过 **ST-Conv Block**（时间卷积 → 图卷积 → 时间卷积）联合建模时空依赖，摒弃 RNN 以提升训练并行度与稳定性。

- **空间依赖**：谱图卷积（Chebyshev/一阶近似）
- **时间依赖**：因果 1D 卷积 + 门控单元（GLU）
- **结构**：多层 ST-Conv Block 堆叠 + 输出层线性映射  
- **图示**：见 `docs/assets/stgcn.svg`

**优点**：参数量低、训练速度快、作为强基线表现稳健。  
**局限**：未显式融合外部因素，对突发性/非平稳变化的表达力有限。

---

## 3. STDF（2020）：多源时空数据融合的通用框架

**要点**：提出 **SpatioTemporal Data Fusion (STDF)**，将数据划分为两路：
1) **直接相关数据**（速度/流量/密度等）→ GCN 主干；  
2) **间接相关数据**（天气、POI、节假日、事件等）→ **SETON** 编码器。

- **SETON**（Spatial Embedding by Temporal convolutiON）：用共享的空间/时间嵌入对外部数据同时编码，参数高效。
- **特征变换器（Feature Transformer）**：将外部嵌入映射到与 GCN 特征相容的空间。
- **融合模块**：与 GCN 中间层特征做细粒度融合。
- **图示**：见 `docs/assets/stdf.svg`

**优点**：融合异构多源信息、缓解维度爆炸与过拟合，实验显示 RMSE 有显著下降。  
**局限**：结构较复杂，需良好的数据治理与特征规范化。

---

## 4. AST-GCN（2021）：属性增强的时空图卷积网络

**要点**：设计 **属性增强单元（A-Cell）**，把**静态属性**（如 POI 分布、道路类型）与**动态属性**（如天气、节假日）分别处理后与交通特征在时间维进行**拼接增强**，再送入 **GCN+GRU**。

- **空间依赖**：GCN  
- **时间依赖**：GRU（结合门控机制捕捉长期依赖）
- **属性建模**：静态 \(S\in\mathbb{R}^{N\times p}\)，动态 \(D\in\mathbb{R}^{N\times w(m+1)}\) 的滑窗累积
- **图示**：见 `docs/assets/astgcn.svg`

**优点**：处理外部因素更细致，解释性较好。  
**局限**：属性通道容易膨胀，质量和覆盖度决定上限。

---

## 5. 复杂度与工程性对比

| 模型 | 时间建模 | 空间建模 | 外部数据 | 参数与效率 | 工程复杂度 |
|---|---|---|---|---|---|
| STGCN | CNN (GLU) | 谱域 GCN | 无 | ★★★★☆（快） | ★★☆☆☆ |
| STDF | CNN/GCN | GCN | SETON+融合 | ★★★☆☆ | ★★★★☆ |
| AST-GCN | GRU | GCN | 静/动态属性增强 | ★★☆☆☆ | ★★★☆☆ |

---

## 6. 何时选哪一个？

- **数据稀缺/仅历史序列**：先上 **STGCN** 作为强基线。  
- **外部因素丰富（天气/POI/事件）**：优先 **STDF**，融合策略稳健且参数友好。  
- **外部属性维度适中且需可解释性**：选择 **AST-GCN**（静/动态属性分治）。

---

## 7. 参考实现与接口约定

仓库 `src/models/` 提供三个模型的 PyTorch 骨架，均遵循：

```python
y = model(x, ext=None, static=None, dynamic=None)
# x: [B, C, N, T]
# ext: 外部嵌入（STDF）
# static/dynamic: 静/动态属性（AST-GCN）
```

---

## 8. 参考文献

[1] **STGCN (2018)** — *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*.  
[2] **STDF (2020)** — *Spatiotemporal Data Fusion in Graph Convolutional Networks for Traffic Prediction*.  
[3] **AST-GCN (2021)** — *Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting*.

