---
name: Subfield Notation Conventions
description: Notation patterns used across ML subfields encountered in summarized papers
type: reference
---

## Conformal Prediction (general)

- $\alpha$: target miscoverage rate (e.g., 0.1 for 90% coverage)
- $1 - \alpha$: desired marginal coverage guarantee
- $s(\boldsymbol{x}, y)$: conformity score (higher = more conforming); non-conformity defined as the negative
- $\hat{q}$: empirical quantile of calibration scores, defined as $\text{Quantile}\!\left(\frac{\lceil(n+1)\alpha\rceil}{n}; \{s(\boldsymbol{x}_i, y_i)\}_{i=1}^n\right)$
- $\mathcal{C}_\alpha(\boldsymbol{x}_{n+1}) = \{y : s(\boldsymbol{x}_{n+1}, y) \geq \hat{q}\}$: prediction set at significance level $\alpha$
- APS score: $s(\boldsymbol{x}, y) := -(\rho(\boldsymbol{x}, y) + u \cdot \pi(\boldsymbol{x})_y)$ where $\rho(\boldsymbol{x},y) = \sum_{c=1}^K \pi(\boldsymbol{x})_c \mathbb{1}[\pi(\boldsymbol{x})_c > \pi(\boldsymbol{x})_y]$
- RAPS score: APS plus rank penalty $\nu \max(o(\boldsymbol{x}, y) - k, 0)$
- Coverage is Beta-distributed: $\mathbb{P}[y_{n+1} \in \mathcal{C}_\alpha(\boldsymbol{x}_{n+1})] \sim \text{Beta}(n+1-l, l)$ where $l = \lfloor(n+1)\alpha\rfloor$

## Conformal Prediction on Graphs — DAPS (Zargarbashi et al., ICML 2023)

- $G = (\boldsymbol{X}, \boldsymbol{A})$: graph with node feature matrix $\boldsymbol{X}$ and adjacency matrix $\boldsymbol{A}$
- $\mathcal{V}_l, \mathcal{V}_u$: labeled and unlabeled node sets; $\mathcal{V}_d$: development (train+val); $\mathcal{V}_c$: calibration
- $\pi(G) = \boldsymbol{\Pi} \in \triangle^{|\mathcal{V}| \times K}$: GNN output; row $v$ is predicted label distribution for node $v$
- $s(v, y) = \boldsymbol{\Pi}_{vy}$: per-node conformity score for candidate label $y$
- $\boldsymbol{H} \in \mathbb{R}^{|\mathcal{V}| \times K}$: matrix of all node-wise scores for all classes
- **DAPS 1-hop diffused score**: $\hat{s}(v, y) = (1-\lambda)s(v,y) + \frac{\lambda}{|\mathcal{N}_v|}\sum_{u \in \mathcal{N}_v} s(u, y)$
- Matrix form: $\hat{\boldsymbol{H}} = (1-\lambda)\boldsymbol{H} + \lambda \boldsymbol{D}^{-1}\boldsymbol{A}\boldsymbol{H}$, $\boldsymbol{D}$ = degree matrix
- $\lambda \in [0,1]$: diffusion parameter; tuned on held-out calibration split; empirically optimal ~0.5
- **Score Propagation (SP) variant**: iterative $\hat{\boldsymbol{H}}^{(t)} = (1-\lambda)\hat{\boldsymbol{H}}^{(0)} + \lambda\hat{\boldsymbol{H}}^{(t-1)}\boldsymbol{D}^{-1}\boldsymbol{A}$; closed form $(1-\lambda)(\boldsymbol{I} - \lambda\boldsymbol{D}^{-1}\boldsymbol{A})^{-1}\boldsymbol{H}$
- NAPS (Clarkson 2022): competing method using weighted CP with adjacency weights; fails on sparse graphs where test nodes have no calibration neighbors (up to 79% of nodes on CoraML with default splits)

## Graph Neural Networks

- $G = (\boldsymbol{X}, \boldsymbol{A})$ or $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: graph with node features and adjacency
- $\mathbf{h}_v^{(k)}$: hidden representation of node $v$ at layer $k$
- $\mathcal{N}(v)$ or $\mathcal{N}_v$: neighborhood of node $v$
- Transductive setting: all nodes observed at training time; only labels of $\mathcal{V}_d$ revealed
- Permutation equivariance: key property that preserves exchangeability of node scores under DAPS
