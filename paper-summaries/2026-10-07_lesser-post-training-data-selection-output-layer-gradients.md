# LESSER: Post-Training Data Selection with Output-Layer Gradients

**arXiv:** 2610.03702  
**Authors:** Lyuxin David Zhang, Eric Wong, Surbhi Goel, Anton Xue  
**Affiliation:** University of Pennsylvania  
**Date:** October 2, 2026

---

## One-Line Summary

Output-layer gradients are a 9.7× cheaper proxy for full-parameter gradients in post-training data selection, preserving batch-level alignment while reducing compute for both SFT and RL data selection.

---

## Problem

The choice of which data to use for post-training large language models substantially affects downstream task performance. Gradient-based data selection has emerged as a principled approach: it ranks candidate training samples by how well their parameter gradients align with the gradients of a small held-out validation set. However, full-parameter gradient computation requires a backward pass through the entire model for every candidate sample in a potentially large pool, making this approach computationally intractable at scale.

---

## Why Existing Approaches Fall Short

- **Full-gradient cost:** Computing the gradient with respect to all model parameters for each data point in a large candidate pool requires one full backward pass per sample. For models with billions of parameters and pools of tens of thousands of candidates, this is prohibitively expensive.
- **Embedding-layer proxies are approximate:** Prior work such as LESS used low-rank approximations of embedding-layer gradients, achieving efficiency but with non-trivial approximation error and additional hyperparameter choices (LoRA rank, projection dimension).
- **Batch-level vs.\ sample-level alignment:** Existing work implicitly assumes that sample-level gradient ranking preserves batch-level quality. LESSER examines whether this assumption is necessary for effective data selection.

---

## Core Method

LESSER (which stands for Last-layer Efficient Sample Selection with Output-layer gRadients) restricts gradient computation to the **output layer** of the language model head—the final linear projection mapping hidden states to vocabulary logits. This is a tiny fraction of total model parameters but captures the gradient signal most directly tied to the training loss.

The method is implemented as a **drop-in wrapper** for any gradient-based data selection algorithm: replace the gradient-extraction step with an output-layer-only backward pass, leaving the selection algorithm itself unchanged.

**Key empirical observation:** Even when output-layer and full gradients rank individual samples differently, the batches they ultimately select have well-aligned gradient directions. This batch-level alignment is the operative quantity for data selection quality.

---

## Technical Formulation

Let $\theta$ denote all model parameters and $\theta_L$ the output-layer parameters. For a data point $x$ and validation set $\mathcal{V}$, gradient-based selection scores:
$$s(x) = \langle \nabla_\theta \ell(x, \theta),\ \nabla_\theta \mathcal{L}(\mathcal{V}, \theta) \rangle$$

LESSER replaces $\theta$ with $\theta_L$:
$$s_\text{LESSER}(x) = \langle \nabla_{\theta_L} \ell(x, \theta),\ \nabla_{\theta_L} \mathcal{L}(\mathcal{V}, \theta) \rangle$$

The computational saving comes from computing only the last-layer Jacobian $\frac{\partial \ell}{\partial \theta_L}$, which requires only a single matrix multiplication involving the final hidden state $h$ and the loss gradient $\frac{\partial \ell}{\partial \text{logits}}$:
$$\nabla_{\theta_L} \ell = \frac{\partial \ell}{\partial \text{logits}} \cdot h^\top$$

This eliminates backpropagation through all intermediate transformer layers.

**Batch alignment theorem (empirical):** Let $B_\text{full}$ and $B_\text{LESSER}$ be the top-$k$ batches selected by full and output-layer gradients respectively. The paper demonstrates empirically that:
$$\left\langle \nabla_\theta \mathcal{L}(B_\text{LESSER}, \theta),\ \nabla_\theta \mathcal{L}(\mathcal{V}, \theta) \right\rangle \approx \left\langle \nabla_\theta \mathcal{L}(B_\text{full}, \theta),\ \nabla_\theta \mathcal{L}(\mathcal{V}, \theta) \right\rangle$$

---

## Learning or Inference Procedure

1. **Compute validation gradient:** Run a forward-backward pass on the held-out validation set using only output-layer parameters. Cache the validation gradient $g_\mathcal{V} = \nabla_{\theta_L} \mathcal{L}(\mathcal{V}, \theta)$.

2. **Score candidates:** For each candidate data point $x_i$, run a forward pass to obtain hidden states, then compute the output-layer gradient $g_i = \nabla_{\theta_L} \ell(x_i, \theta)$ with a single matrix multiplication. Compute alignment score $s_i = \langle g_i, g_\mathcal{V} \rangle$.

3. **Select data:** Rank candidates by $s_i$ and select the top-$k$ points. Use these as the post-training data.

4. **Post-train:** Fine-tune the model on the selected data using the standard SFT or RL training objective.

---

## What the Guarantee Says

The paper does not provide a formal guarantee that output-layer gradient alignment implies full-gradient alignment at the sample level. However, it provides strong empirical evidence that **batch-level** alignment—the operative quantity for gradient-based selection—is preserved. The theoretical intuition is that the last layer must aggregate information from all earlier layers, so its gradient captures the dominant component of the full-gradient direction along the data manifold.

---

## Experimental Findings

- **FLOP reduction:** LESSER reduces feature-extraction cost by **9.7×** for SFT selection and **3.0×** for RL selection benchmarks relative to full-gradient methods.
- **Performance parity:** On downstream task evaluations, LESSER matches full-gradient selection performance within noise across all tested settings.
- **Sample vs.\ batch alignment:** Output-layer gradients rank individual samples differently from full gradients (Spearman correlation varies), but the batches selected by both approaches have nearly identical gradient alignment with the validation set.
- **Compatibility:** LESSER wraps existing selection methods (LESS, DataInf, etc.) without modification to the selection algorithm itself.

---

## Ablations and Interpretation

**Layer depth:** Ablations over which layer's gradients to use show a monotone trend: deeper layers provide better proxies for full gradients, with the output layer being best. Mid-layer gradients also reduce cost but with a larger quality gap.

**Validation set size:** Performance of LESSER degrades gracefully as the validation set shrinks; as few as 128 validation examples suffice for reliable selection.

**SFT vs.\ RL:** The FLOP reduction is larger for SFT (9.7×) than for RL (3.0×) because RL requires computing gradients over longer generated sequences, making each full backward pass more expensive and the relative saving of the output-layer approximation less dramatic.

---

## Reference

Lyuxin David Zhang, Eric Wong, Surbhi Goel, Anton Xue.
**LESSER: Post-Training Data Selection with Output-Layer Gradients.**
arXiv:2610.03702, October 2026.
https://arxiv.org/abs/2610.03702
