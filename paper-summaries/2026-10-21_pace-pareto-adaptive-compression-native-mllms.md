# PACE: Pareto-Adaptive Compression for Efficient Native MLLMs

**arXiv:** 2610.24005  
**Venue:** NeurIPS 2026  
**Authors:** Lei Tian, Chunzheng Zhu, Kai Wu, Yifan Zhang, Haihua Yang  
**Date:** October 15, 2026

---

## One-Line Summary

PACE introduces Pareto-aware visual token compression for native (ViT-free) MLLMs using orthogonal spatial saliency decomposition and multi-objective budget allocation, achieving near-lossless or improved performance at 25–50% token retention with 2× FLOPs reduction.

---

## Problem

Native multimodal large language models (MLLMs)—those that process raw image patches directly without a separate vision encoder such as a ViT—are becoming increasingly common because they reduce architectural complexity and enable end-to-end optimization. However, native architectures also mean that the model's attention mechanism must process a large number of visual tokens that grow quadratically in sequence length. Existing visual token compression methods were designed for ViT-based pipelines, where the encoder's attention pattern can be exploited as a saliency signal. Native MLLMs have no such encoder, making prior methods inapplicable.

---

## Key Observation: ViT-Free Models Tolerate Aggressive Compression

The authors observe that native MLLMs tolerate aggressive visual token compression with minimal or even improved performance—but only when the compression method respects the spatial layout of the image and balances efficiency gains against spatial coverage, rather than selecting tokens purely by a per-token attention score.

---

## Why Existing Approaches Fall Short

- **ViT-based compression methods (FastViT, ToMe, etc.):** Rely on the ViT's [CLS] attention or patch-level similarity, which are not available in native architectures.
- **Uniform subsampling:** Discards spatial structure; misses locally salient regions that happen to be scattered across the image.
- **Greedy per-token attention thresholding:** Optimizes a single objective (per-token attention score) without considering how the selected tokens cover the image spatially or how well the budget serves downstream benchmarks.

---

## Core Method

**Orthogonal spatial saliency decomposition:**
PACE splits the model's self-attention into two orthogonal components—one along the height axis and one along the width axis—to compute separate saliency estimates. This is then gated by a positional attention signal that encodes where in the image the current query is attending. The result is a 2D saliency map that captures both global and local salient regions without relying on a ViT encoder.

**Pareto budget allocation:**
Rather than choosing a single compression ratio, PACE formulates token budget allocation as a Pareto optimization problem over three objectives:
1. **Performance preservation:** minimize drop in task accuracy relative to full-token baseline.
2. **Inference efficiency:** maximize FLOPs reduction.
3. **Spatial coverage:** ensure that selected tokens span the image without large gaps.

A learned allocation head solves for the Pareto frontier at inference time, selecting the operating point for each input.

---

## Technical Formulation

Let $\mathbf{A} \in \mathbb{R}^{N \times N}$ be the attention matrix for visual tokens of sequence length $N$. PACE decomposes it as:

$$\mathbf{A} = \mathbf{A}_H \otimes \mathbf{A}_W + \epsilon$$

where $\mathbf{A}_H \in \mathbb{R}^{h \times h}$ and $\mathbf{A}_W \in \mathbb{R}^{w \times w}$ are row-wise and column-wise marginals (with $N = h \times w$), and $\epsilon$ captures the residual.

The saliency score for token $(i,j)$ is:

$$s_{ij} = \alpha \cdot [\mathbf{A}_H]_{ii} + \beta \cdot [\mathbf{A}_W]_{jj} + \gamma \cdot p_{ij}$$

where $p_{ij}$ is the positional attention signal and $(\alpha, \beta, \gamma)$ are learned scalars.

The Pareto allocation objective is:

$$\min_\mathbf{m} \quad \left(\text{PerfDrop}(\mathbf{m}),\ -\text{FLOPs\_red}(\mathbf{m}),\ -\text{SpatCov}(\mathbf{m})\right)$$

subject to $\|\mathbf{m}\|_0 = B$ (budget $B$ tokens), solved approximately by a small allocation network trained on a performance-efficiency frontier constructed during training.

---

## Learning or Inference Procedure

1. **Construct training frontier:** Run the model at several compression ratios; record task performance, FLOPs, and spatial coverage.
2. **Train allocation network:** A small MLP maps input complexity features to a budget $B$ and a weighting $(\alpha, \beta, \gamma)$ on the Pareto frontier.
3. **At inference:** The allocation network selects $B$ and computes saliency scores; the top-$B$ tokens by score are retained; the remaining tokens are pruned before the model's subsequent layers.
4. No modification to the base MLLM is required; PACE is a plug-in compression layer.

---

## What the Paper Claims

- **Performance:** Near-lossless task accuracy at 25–50% token retention; improvement on some benchmarks (visual question answering, chart understanding).
- **Efficiency:** >2× FLOPs reduction vs. full-token baseline at 50% retention.
- **Generality:** Works across several native MLLM architectures without architecture-specific tuning.

---

## Experimental Findings

- **Models tested:** Multiple native MLLM architectures (ViT-free design).
- **Benchmarks:** Visual question answering (VQA), chart understanding, spatial reasoning, multimodal reasoning.
- **At 50% token retention:** Near-lossless performance; FLOPs reduced by 2.1–2.4×.
- **At 25% token retention:** Performance drops slightly below baseline on spatial-reasoning tasks but improves on chart understanding (spatial coverage objective prevents missing chart regions).
- **Ablations confirm** that the positional attention signal contributes most to spatial-reasoning tasks; removing it hurts.

---

## Ablations and Interpretation

- **Without Pareto allocation (fixed ratio):** Setting a global 50% budget works well on average but fails on inputs that are either information-sparse (wasteful) or information-dense (lossy).
- **Without orthogonal decomposition (scalar attention):** Captures global attention but misses fine-grained spatial structure; saliency map is blurry.
- **Height-only or width-only saliency:** Misses salient regions that are localized in the orthogonal direction.
- **Why native MLLMs tolerate compression better:** The authors hypothesize that without a ViT pre-processing stage, the model spreads attention more uniformly early and learns to aggregate redundant tokens internally; compression makes this implicit process explicit.

---

## Reference

Lei Tian, Chunzheng Zhu, Kai Wu, Yifan Zhang, Haihua Yang. **PACE: Pareto-Adaptive Compression for Efficient Native MLLMs**. arXiv:2610.24005. NeurIPS 2026.  
https://arxiv.org/abs/2610.24005
