# Spectral Degeneration of Self-Attention under Isotropic Score Geometry

**Authors:** Mengda Li, Jianfeng Yao  
**arXiv:** 2611.02778  
**Categories:** cs.LG, stat.ML  
**Venue:** NeurIPS 2026  
**Affiliation:** The Chinese University of Hong Kong, Shenzhen  
**Submitted:** November 2, 2026  
**URL:** https://arxiv.org/abs/2611.02778

---

## Headline

Under an idealized isotropic geometry for the pre-softmax score matrix, the softmax nonlinearity causes attention to degenerate spectrally: dominant modes are amplified while the bulk collapses to zero, with causal attention showing logarithmically growing stable rank versus bounded stable rank for bidirectional attention.

## Key Findings

- The softmax step creates a sharp spectral split: dominant eigenvalues are amplified while the remaining bulk collapses, explaining the empirically observed low-rank structure of attention matrices.
- For bidirectional (encoder) attention, the stable rank of the attention matrix remains bounded as sequence length L increases.
- For causal (decoder) attention, the stable rank grows as O(log L), reflecting the asymmetric weighting from causal masking.
- These results provide theoretical justification for low-rank attention approximation methods and suggest that causal models may require different regularization than bidirectional ones.

## Problem

Empirical observations consistently show that attention matrices in trained Transformers are approximately low-rank, and that this low-rank structure emerges across architectures and datasets. Several empirical works have proposed low-rank approximations of attention (e.g., Linformer, Nystromformer), but the theoretical reasons for why full-rank attention heads degenerate toward low rank through training remain unclear. This paper asks: is there a spectral mechanism intrinsic to the softmax attention computation that drives this degeneration, independent of learning dynamics?

## Setup and Assumptions

The paper analyzes the attention matrix A = softmax(S/sqrt(d)) where S is the pre-softmax score matrix. The key simplifying assumption is that the score matrix S has isotropic geometry: the query-key products are drawn from a Gaussian random matrix model with identity covariance in the pre-softmax space. This models a "generic" attention head before specialization, isolating the effect of softmax from learning.

Under this model, random matrix theory gives precise characterizations of the spectral distribution of S, which the paper then propagates through the softmax function.

## Main Theoretical Results

**Spectral bulk collapse:** The bulk of the eigenspectrum of A concentrates near zero as the head dimension d grows, while a small number of dominant eigenvalues remain bounded away from zero. This is a direct consequence of the exponential function in softmax amplifying large scores while suppressing small ones.

**Bidirectional attention (bounded stable rank):** For bidirectional attention, the stable rank — defined as ||A||²_F / ||A||²_2 — converges to a bounded constant as L → ∞. This means the effective rank of the attention pattern does not grow with sequence length.

**Causal attention (logarithmic stable rank growth):** For causal attention, the asymmetric structure of the mask introduces a gradient in the per-row normalizations of softmax. The analysis shows this causes the stable rank to grow as O(log L), meaning causal attention heads become "more low-rank" relative to their size more slowly than bidirectional heads.

## Technical Approach

The analysis uses tools from free probability theory and the theory of Wigner matrices to characterize the eigenspectrum of exp(S), then tracks how the row-normalization step of softmax modifies this. The key technical lemma shows that the softmax row-normalization can be approximated by a rank-1 update to exp(S) in the high-dimensional limit, and this update concentrates all the spectral mass that was removed from the bulk into a small number of dominant modes.

## Empirical Validation

The authors validate their theoretical predictions against:
- Pretrained GPT-2 and BERT attention heads, showing that the stable rank distributions match the theoretical predictions qualitatively.
- Synthetic random attention heads under the isotropic score model, confirming tight concentration bounds.

## Implications

- **Low-rank attention compression:** The theoretical bound on stable rank gives a principled justification for why rank-k attention approximations (with k independent of L) lose little information for bidirectional models.
- **Causal vs. bidirectional design:** The O(log L) growth for causal models suggests that compression ratios for decoder-only models should scale mildly with sequence length, while encoder models can use fixed compression ratios.
- **Regularization:** Encouraging isotropic score geometry during training (e.g., via spectral normalization of Q and K) may help prevent collapse.
