# CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference

**arXiv:** 2609.26300  
**Authors:** Anonymous et al.  
**Affiliation:** Not publicly available  
**Date:** September 2026

---

## One-Line Summary

By pairing block-level KV selection with mean compensation — which admits an exact log-partition identity — CompKV derives a principled selection priority that jointly accounts for block attention mass and within-block logit variation, beating prior eviction and selection methods on long-context benchmarks.

---

## Problem

Long-context LLM inference is memory-bound: fitting 128K-token contexts requires gigabytes of KV cache that exceed GPU memory. Sparse-attention methods reduce memory by loading only a subset of KV blocks at each layer, but existing selection heuristics (top-K by attention mass, recency, or activation norms) discard unselected context entirely. This introduces approximation error that grows with context length and disproportionately hurts tasks that require global evidence aggregation.

---

## Why Existing Approaches Fall Short

- **Eviction methods** (H2O, StreamingLLM) permanently remove tokens from the cache and cannot recover evidence from evicted positions.
- **Selection-only sparse attention** loads the top-K blocks per query but treats unselected blocks as zero, a poor approximation when those blocks carry non-trivial attention mass.
- **Selection + zero-padding** does not preserve the softmax normalization: the partition function is computed only over selected tokens, biasing attention distributions.
- **KV quantization and offloading** compress or move the cache but do not reduce the number of tokens read per attention call.

None of these methods account for the distributional effect of unselected tokens on the attention output.

---

## Core Method

**Mean Compensation.** CompKV replaces every logit in an unselected block with its block mean rather than zero. This choice is not arbitrary: the block-mean approximation yields an exact identity for the log-partition function. Specifically, if the true logit for an unselected block has values $\{e_1, \ldots, e_B\}$, replacing each with $\bar{e} = \frac{1}{B}\sum_i e_i$ exactly preserves the block's contribution to the softmax denominator (up to a first-order correction). This means the attention denominator is correctly computed even though individual within-block attention weights are approximated.

**Compensation-Aware Selection Criterion.** Given mean compensation, the residual error introduced by approximating an unselected block depends on:
1. The block's total attention mass (how much it contributes to the output).
2. The within-block logit variation (how much individual weights differ from the block mean).

CompKV derives an optimal selection priority as the product of these two factors. Blocks with high mass but low variation are adequately covered by mean compensation and can safely remain unselected; blocks with high mass and high variation are prioritized for selection.

---

## Technical Formulation

Let $\mathbf{Q} \in \mathbb{R}^{T_q \times d}$ be the query and let the KV cache be partitioned into $B$ blocks of size $b$. Define the unnormalized attention logits for block $j$ as $\mathbf{e}_j \in \mathbb{R}^{T_q \times b}$.

Under mean compensation, the approximated log-partition function for block $j$ is:
$$\log Z_j^{\text{approx}} = \log\!\left(b \cdot \exp\!\left(\bar{e}_j\right)\right) = \bar{e}_j + \log b$$
which equals the exact $\log Z_j = \log\!\sum_{i=1}^{b} \exp(e_{j,i})$ when $e_{j,i} = \bar{e}_j$ for all $i$.

The residual error from mean compensation in block $j$ is bounded by:
$$\Delta_j \leq m_j \cdot \sigma_j$$
where $m_j = \text{softmax\_mass}(\mathbf{e}_j)$ is the block's attention mass and $\sigma_j = \text{std}(\mathbf{e}_j)$ is the within-block logit standard deviation.

The CompKV selection priority score is:
$$s_j = m_j \cdot \sigma_j$$
Blocks with the highest $s_j$ are selected; the rest are replaced by their mean.

---

## Learning or Inference Procedure

CompKV is a **training-free inference-time method**. At each attention layer and query step:

1. **Block statistics computation:** A fused GPU kernel computes per-block attention mass estimates and logit standard deviations using a grouped variance projection of the current query against pre-computed block mean keys.
2. **Selection:** Blocks are ranked by $s_j = m_j \cdot \sigma_j$; the top-$K$ blocks are selected for exact attention.
3. **Compensation:** For unselected blocks, mean compensation is applied — the block mean key/value is used and the partition function is adjusted accordingly.
4. **Output computation:** Standard attention is computed over selected blocks; compensated contributions from unselected blocks are added to the denominator.

**Implementation details:** The complete BF16 KV cache is kept in pinned CPU memory alongside FP32 block summaries. A main CUDA stream handles selection-region transfer and exact-attention kernels; an auxiliary stream handles mean-value transfers in parallel, eliminating blocking synchronization.

---

## What the Guarantee Says

The mean-compensation scheme guarantees that the log-partition function is preserved exactly (to first order) for each unselected block when all within-block logits are equal. For non-uniform blocks, the residual error is bounded proportionally to $m_j \cdot \sigma_j$, which is precisely the quantity CompKV uses to prioritize selection. This gives a principled worst-case guarantee: the total approximation error is minimized under the fixed-$K$ selection budget by using the derived priority scores.

---

## Experimental Findings

Evaluated on three models — Llama-3.1-8B-Instruct, Qwen3-8B, and Qwen3-32B — with context lengths up to 128K tokens on:

- **RULER:** Controlled retrieval, tracing, and aggregation tasks designed to stress long-context capabilities under tight exact-read budgets.
- **LongBench-Pro:** Realistic tasks requiring localized evidence retrieval or global context integration.

CompKV consistently outperforms:
- Standard sparse attention (top-K by attention mass only)
- Eviction-based methods (H2O, StreamingLLM variants)
- Zero-compensation sparse attention

The gains are largest on aggregation tasks (where global context matters) and at tighter budgets (fewer selected blocks), confirming that compensation matters most when selection must be aggressive.

---

## Ablations and Interpretation

- **Selection criterion ablation:** Replacing $s_j = m_j \cdot \sigma_j$ with mass-only ($m_j$) or variation-only ($\sigma_j$) degrades performance, confirming the joint criterion is necessary.
- **Compensation type:** Mean compensation outperforms zero compensation and approximates full attention more closely, especially for high-variation blocks.
- **Block size:** Larger blocks increase per-block variation and make compensation more useful; smaller blocks reduce variation and reduce the benefit.
- **CPU offloading overhead:** The fused kernel architecture largely hides data-transfer latency by overlapping transfers with GPU computation, keeping throughput competitive with GPU-only sparse attention at high batch sizes.

---

## Reference

Anonymous et al. **CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference**. arXiv:2609.26300, September 2026.  
https://arxiv.org/abs/2609.26300
