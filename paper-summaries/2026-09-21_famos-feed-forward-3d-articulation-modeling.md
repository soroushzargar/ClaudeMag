# FAMOS: Feed-Forward 3D Articulation Modeling from Sparse Observations

**arXiv:** 2609.20817  
**Submitted:** September 17, 2026  
**Authors:** Kevin Qu, Stefan Ainsworth, Jiahao Wang, Gordon Wetzstein, Qianqian Wang  
**Affiliation:** Stanford University; ETH Zürich  
**Venue:** arXiv preprint

---

## Headline Finding

FAMOS is a feed-forward model that predicts movable-part segmentation and joint parameters for articulated objects from a sparse, unordered set of partial point clouds, outperforming all prior feed-forward and optimization-based baselines on PartNet-Mobility, ACD, and ArtiCraft-10K by up to 27.8 points, by jointly reasoning over multi-view articulation evidence through a Multi-state Articulation Transformer and an Observed Articulation Span training objective.

---

## Key Findings (Pyramid Layer 2)

1. **Single-view priors are insufficient for articulation.** Existing feed-forward methods infer articulation from a single observation and thus lean heavily on learned category-level shape priors. This fails on novel object instances, unusual joint configurations, and object categories under-represented in training data. Multi-view observations contain rich complementary information about both geometry and motion range that single-view methods discard.

2. **Multi-state Articulation Transformer (MAT) aggregates cross-observation cues.** FAMOS introduces a transformer architecture with alternating state-wise attention (attending across part queries within a single observation) and global attention (attending across all observations simultaneously). This allows the model to integrate articulation evidence that may only appear in a subset of views (e.g., a drawer partially open in one image, fully closed in another).

3. **Observed Articulation Span (OAS) objective exploits motion diversity.** The OAS objective supervises the motion range each part actually exhibits across the input observation set. By requiring the model to predict how far each part moves across the provided views, rather than just its pose in each view independently, OAS trains the model to exploit the full multi-view signal and infer plausible joint limits.

4. **Procedural synthesis overcomes dataset scarcity.** Existing articulated object datasets (PartNet-Mobility, etc.) are too small and category-limited to train expressive multi-view models. FAMOS introduces a procedural data generator that synthesizes fully annotated articulated assets on the fly during training, providing effectively unlimited diversity of geometries, joint types, and part configurations.

---

## Methodology (Pyramid Layer 3)

### Background: Articulated Object Understanding

An articulated object consists of $M$ rigid parts $\{P_m\}_{m=1}^M$ connected by kinematic joints. The joint parameters include the joint type (revolute or prismatic), joint axis $\mathbf{a} \in \mathbb{R}^3$, joint origin $\mathbf{o} \in \mathbb{R}^3$, and motion range $[\theta_{\min}, \theta_{\max}]$. Given a set of partial point cloud observations $\{X_n\}_{n=1}^N$ of the same object in different configurations (states), the goal is to predict per-point part assignments and per-part joint parameters.

### Multi-state Articulation Transformer

FAMOS represents each observation $X_n$ as a set of point features obtained by a shared PointNet-style encoder $\phi$. Each observation produces a feature map $F_n = \phi(X_n) \in \mathbb{R}^{T \times D}$ where $T$ is the number of point tokens and $D=768$. The model maintains $K=16$ learnable part query vectors $\mathbf{q}_k \in \mathbb{R}^D$.

The MAT stacks $L=6$ layers, each alternating between:

- **State-wise attention:** For each observation $n$, the part queries cross-attend to the point features of that observation:
$$\mathbf{q}_k^{(n)} \leftarrow \text{Attn}(\mathbf{q}_k, F_n)$$

- **Global attention:** Part queries from all observations attend jointly, enabling the model to compare articulation evidence across states:
$$\{\mathbf{q}_k^{(n)}\}_{n=1}^N \leftarrow \text{Attn}\!\left(\{\mathbf{q}_k^{(n)}\}_{n=1}^N, \{\mathbf{q}_k^{(n)}\}_{n=1}^N\right)$$

After $L$ layers, the per-part queries $\{\mathbf{q}_k\}$ are decoded by separate MLP heads into joint axis, origin, motion range, and part membership logits for each input point.

### Observed Articulation Span Objective

Let $\theta_n^{(m)}$ be the joint angle of part $m$ in observation $n$. The observed span is:
$$\Delta_m = \max_n \theta_n^{(m)} - \min_n \theta_n^{(m)}$$

FAMOS adds a supervised regression head that predicts $\hat{\Delta}_m$ from the part query $\mathbf{q}_k$:
$$\mathcal{L}_{\text{OAS}} = \sum_m \left(\hat{\Delta}_m - \Delta_m\right)^2$$

This forces the model to summarize the motion evidence seen across all observations into a meaningful span estimate, encouraging the global attention layers to aggregate cross-state information rather than process each view independently.

### Procedural Data Generation

The procedural generator samples random kinematic trees (depth 1–3), assigns primitive or CAD-derived meshes to each link, samples random joint axes and ranges, and renders point clouds from multiple configurations with random sensor noise and partial occlusions. Approximately 50,000 synthetic assets are generated per training run, each yielding 4–8 observations.

---

## Technical Formulation

The full training loss combines part segmentation, joint parameter regression, and OAS:

$$\mathcal{L} = \lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{axis}}\mathcal{L}_{\text{axis}} + \lambda_{\text{origin}}\mathcal{L}_{\text{origin}} + \lambda_{\text{range}}\mathcal{L}_{\text{range}} + \lambda_{\text{OAS}}\mathcal{L}_{\text{OAS}}$$

Joint axis regression uses a cosine loss:
$$\mathcal{L}_{\text{axis}} = 1 - \left|\cos\langle \hat{\mathbf{a}}_m, \mathbf{a}_m \rangle\right|$$

Part segmentation uses a Hungarian-matched cross-entropy between predicted membership logits and ground-truth assignments.

---

## Learning or Inference Procedure

1. Encode each input observation $X_n$ independently with the PointNet encoder to get $F_n$.
2. Initialize $K$ learnable part query vectors.
3. Run $L$ MAT layers alternating state-wise and global attention.
4. Decode part queries into joint parameters and point-wise part logits.
5. At inference: the OAS head is not used. Part assignments and joint parameters are read directly from the decoder outputs.
6. The model supports any $N \geq 1$ observations; for $N=1$ it reduces to a single-view baseline.

---

## What the Guarantee Says

FAMOS does not provide a formal approximation guarantee. The paper shows empirically that as $N$ increases from 1 to 4, articulation accuracy improves monotonically across all three benchmarks, with the largest gain between 1 and 2 observations. This validates the design choice of multi-view aggregation over single-view inference.

---

## Experimental Findings

### Benchmarks
- **PartNet-Mobility:** 45 categories, 2,346 objects (standard articulation benchmark)
- **ACD:** Articulated CAD dataset with out-of-distribution CAD geometries
- **ArtiCraft-10K:** Large-scale community-contributed dataset with diverse object types

### Main Results (4 input observations, geodesic axis error in degrees, lower is better)

| Method | PartNet-Mob | ACD | ArtiCraft-10K |
|--------|------------|-----|---------------|
| A-SDF | 24.1 | 31.4 | 38.7 |
| DITTO | 21.3 | 28.6 | 35.2 |
| Particulate | 18.4 | 22.5 | 29.1 |
| **FAMOS** | **10.3** | **(-5.3) 16.9** | **7.6** |

FAMOS outperforms Particulate by 8.1, 5.6, and 21.5 points respectively (and by 27.8 on ACD when measured by full joint parameter accuracy metric).

---

## Ablations and Interpretation

- **Removing OAS objective:** Axis error on PartNet-Mobility increases by 2.9 points; the model fails to distinguish between full and partial motion ranges.
- **Global vs. state-wise only attention:** Using only state-wise attention degrades axis error by 3.6 points, confirming that cross-view comparison is critical.
- **Synthetic data only vs. mixed real/synthetic:** Adding real PartNet-Mobility training data on top of synthetic data improves accuracy by 1.8 points; removing synthetic data causes a 4.2-point drop, confirming the procedural generator's importance.
- **Number of observations N:** Going from N=1 to N=2 gains 4.3 points; from N=2 to N=4 gains another 2.1 points; gains plateau around N=6.

---

## Reference

Kevin Qu, Stefan Ainsworth, Jiahao Wang, Gordon Wetzstein, and Qianqian Wang. **FAMOS: Feed-Forward 3D Articulation Modeling from Sparse Observations.** arXiv:2609.20817, September 2026. https://arxiv.org/abs/2609.20817
