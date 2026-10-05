# TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning

**arXiv:** 2610.02199  
**Authors:** Jichao Jiang, Cristian McGee, El Houcine Bergou, Hanqin Cai, Aritra Dutta  
**Affiliation:** University of Central Florida; Mohammed VI Polytechnic University (UM6P)  
**Date:** October 1, 2026

---

## One-Line Summary

TACO achieves a 174× reduction in optimizer state memory over AdamW8bit and enables full-parameter fine-tuning of 32B-parameter LLMs on a single 80 GB H100, by taking the exact steepest-descent step under a dimension-normalized 1→1 operator norm—selecting the sign of the largest-magnitude column entry—while maintaining comparable downstream accuracy.

---

## Problem

Full-parameter fine-tuning of LLMs is increasingly important for domain adaptation and instruction following, but it is severely constrained by GPU memory. On a single 80 GB H100, AdamW (the standard optimizer) can fine-tune models only up to roughly 7B parameters before hitting memory limits, because it maintains first and second momentum buffers adding 2× the model's parameter count in optimizer state. Parameter-efficient methods (LoRA, prefix tuning) trade expressivity for memory efficiency; they cannot update all parameters and may underfit complex domain adaptation tasks.

---

## Why Existing Approaches Fall Short

- **AdamW and AdaGrad variants** maintain dense first- and second-order moment estimates per parameter, incurring 2–3× model size in optimizer state.
- **8-bit optimizers (AdamW8bit, bitsandbytes)** quantize the moment buffers but still require storing a full first and second moment per parameter.
- **SignSGD and Muon** reduce state memory by discarding second moments, but their update directions are not derived from a principled norm-minimization argument in the column-wise operator sense. Muon uses the Nesterov update with an operator-norm steepest-descent motivation but does not achieve the minimum-state extreme.
- **LoRA and variants** cannot update all parameters; full-rank updates remain out of reach for large models.

---

## Core Method

**Steepest descent under the 1→1 operator norm.** TACO derives its update rule from a first-principles norm-minimization argument. Consider a 2D weight matrix $W \in \mathbb{R}^{m \times n}$. The operator norm $\|G\|_{1 \to 1}$ of the gradient $G$ equals the maximum absolute column sum. The steepest descent direction under a dimension-normalized 1→1 norm constraint selects, for each column $j$, the row index $i^* = \arg\max_i |G_{ij}|$ and sets the update matrix entry to $\text{sign}(G_{i^*j})$, with all other entries in that column set to zero.

This yields an update matrix $U \in \{-1, 0, +1\}^{m \times n}$ with exactly one nonzero entry per column—hence "ternary" (signs $\{-1, +1\}$) and "one-sparse" (per column). The only persistent optimizer state required is the gradient itself (needed for the argmax computation), which is discarded after the update step. No momentum buffers are maintained.

---

## Technical Formulation

Let $G^{(t)}$ be the gradient of the loss with respect to $W$ at step $t$. TACO computes:
$$i^*(j) = \arg\max_{i} |G_{ij}^{(t)}|, \quad \forall j \in [n]$$
$$U_{ij}^{(t)} = \begin{cases} \text{sign}(G_{i^*(j),j}^{(t)}) & \text{if } i = i^*(j) \\ 0 & \text{otherwise} \end{cases}$$
$$W^{(t+1)} = W^{(t)} - \eta \cdot U^{(t)}$$

For non-2D tensors (e.g., embedding tables or 1D biases), TACO falls back to signSGD.

The optimizer state at step $t$ is exactly $\{G^{(t)}\}$, which is discarded immediately after computing $U^{(t)}$. Persistent optimizer state between steps is zero (beyond the model weights themselves), contrasted with AdamW's two persistent moment buffers.

In practice, a single BF16 gradient copy is held in memory during the backward pass; once $U^{(t)}$ is computed and applied, that buffer is freed, yielding negligible persistent state.

---

## Learning or Inference Procedure

**Forward pass:** Standard forward pass; no modification.

**Backward pass:** Compute gradient $G^{(t)}$ for each 2D weight matrix.

**Update:**
1. For each column $j$ of $G^{(t)}$, find $i^*(j) = \arg\max_i |G_{ij}^{(t)}|$.
2. Construct $U^{(t)}$ with ternary one-sparse columns.
3. Apply weight update $W \leftarrow W - \eta U^{(t)}$ in-place (no extra buffer beyond the gradient).
4. Free $G^{(t)}$.

**Learning rate schedule:** Cosine or linear warmup-decay as in standard fine-tuning; no additional hyperparameters beyond $\eta$.

**Memory footprint:** Peak memory during a training step equals model parameters + one BF16 gradient copy; no persistent optimizer buffers between steps.

---

## What the Guarantee Says

Under a convex loss with Lipschitz-continuous gradient, TACO's update rule achieves convergence at rate $O(1/\sqrt{T})$ matching signSGD, with the additional property that the update matrix $U^{(t)}$ is the exact steepest-descent direction under the 1→1 operator norm constraint—meaning no other ternary update achieves a lower objective decrease guarantee under the operator-norm geometry. In practice, the paper empirically demonstrates that TACO matches AdamW accuracy on a range of fine-tuning tasks despite the drastically reduced optimizer state.

---

## Experimental Findings

- **Memory reduction:** Persistent optimizer state reduced from 27.7 GB (AdamW8bit) to 0.16 GB on OPT-13B—a 174× reduction. Peak training memory reduced 2.9× (80.6 GB → 27.5 GB).
- **Large-model access:** TACO enables full-parameter fine-tuning of OPT-30B and LLaMA-32B on a single 80 GB H100 GPU, previously impossible with AdamW or AdamW8bit.
- **Downstream accuracy:** On SuperGLUE, SQuAD v2, and instruction following benchmarks, TACO achieves accuracy within 0.5–1.0% of AdamW across all tested model families (OPT, LLaMA, Mistral).
- **Training throughput:** Runtime per step is within 5% of AdamW8bit due to the simplicity of the argmax-and-sign update.
- **Transferability:** TACO performs consistently across OPT, LLaMA, Mistral, and Phi model families with no architecture-specific hyperparameter tuning.

---

## Ablations and Interpretation

- **Row-wise vs. column-wise argmax:** Applying the argmax row-wise rather than column-wise (violating the operator-norm derivation) slightly degrades accuracy, confirming that the theoretical motivation matters in practice.
- **Top-$k$ per column (TACO-$k$):** Allowing $k > 1$ nonzero entries per column interpolates between TACO and a full dense gradient, gradually recovering AdamW accuracy at the cost of $k$-fold increased optimizer state; $k=1$ is Pareto-optimal for memory-constrained settings.
- **Momentum ablation:** Adding a first-moment buffer (moving average of $U^{(t)}$) modestly improves convergence speed on some tasks but doubles persistent state; TACO without momentum is preferred for memory-constrained regimes.
- **Comparison to LoRA:** At equal GPU memory budget, TACO with full-parameter access outperforms LoRA with comparable rank on domain-specific fine-tuning tasks, confirming the value of full-rank updates.

---

## Reference

Jichao Jiang, Cristian McGee, El Houcine Bergou, Hanqin Cai, Aritra Dutta. **TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning**. arXiv:2610.02199, October 2026.  
https://arxiv.org/abs/2610.02199
