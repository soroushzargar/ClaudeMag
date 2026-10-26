# MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs

**arXiv:** 2610.30206  
**Venue:** NeurIPS 2026  
**Authors:** Minsang Kim, Seungjun Baek  
**Date:** October 2026

---

## One-Line Summary

MoEZip introduces a Fisher-information-based routing sensitivity metric that, combined with attention importance, selects KV cache entries to retain without degrading expert routing in sparse MoE LLMs—outperforming the prior state-of-the-art by 2.3× to 4.9× at 10% retention.

---

## Problem

Sparse mixture-of-experts (MoE) language models—such as Mixtral and DeepSeek-MoE—reduce per-token compute by routing tokens to a small subset of experts. However, this routing is conditioned on hidden states that depend on the full key-value (KV) cache. When KV entries are evicted to reduce memory, the router sees a different attention distribution than it would with the full context, which can change which experts are selected mid-generation. These routing perturbations compound across layers and tokens, degrading generation quality in ways that standard attention-based compression scores—designed for dense transformers—cannot detect or prevent.

---

## Key Observation: Routing Sensitivity Varies Across Context Positions

Not all KV entries affect the routing decision equally. A position that contributes strongly to the attention of routing-relevant heads will, if evicted, push the expert selection probabilities away from their full-context values. By measuring this routing sensitivity per position and per layer, one can preferentially retain entries that jointly satisfy two criteria: they carry high attention weight for instruction-relevant tokens, and their removal would most perturb the router's output distribution.

---

## Why Existing Approaches Fall Short

- **Attention-only scoring (H2O, SnapKV, etc.):** These methods evict entries with low cumulative attention without considering whether those entries influence the MoE router. In dense models this is sound; in MoE models it can silently alter expert dispatch.
- **Magnitude-based pruning:** Low-magnitude entries may still carry routing-critical information if they appear in positions where the router's gate function is sensitive.
- **MoE-agnostic compression pipelines:** Benchmarks such as PiKV and MoE-nD benchmark general composable compression but do not derive importance scores from the routing distribution itself.

---

## Core Method

MoEZip derives an importance score $\phi_t(i)$ for each KV position $i$ at layer $t$ using two signals:

1. **Routing sensitivity via Fisher information:** For a given routing function $r(\cdot)$, the authors derive an approximation of how much the KV eviction of position $i$ would shift the routing distribution. For softmax-type routing ($\text{softmax}(W_r h)$ where $h$ is the post-attention hidden state) and sigmoid-type routing, they derive closed-form approximations of the Fisher information of the routing distribution with respect to the attention output. High Fisher information implies high routing sensitivity.

2. **Attention score of instruction tokens:** The accumulated attention weight that instruction tokens assign to position $i$ captures the position's semantic importance for the current task.

The final score is a weighted combination of these two signals. Positions with both high attention and high routing sensitivity are retained first; positions with low scores in both dimensions are evicted.

---

## Technical Formulation

Let $p_t(i) = \text{softmax}(W_r h_t(i))$ denote the expert routing distribution conditioned on the hidden state $h_t(i)$ produced by attending over position $i$. The Fisher-information-based routing sensitivity for position $i$ is:

$$\mathcal{F}_t(i) = \mathbb{E}_{e \sim p_t(i)}\left[\left(\nabla_{h_t(i)} \log p_t(i)[e]\right)^2\right]$$

For softmax routing this reduces to a closed form involving the routing logits' variance, and for sigmoid routing an analogous derivation holds. Combined with the instruction-token attention weight $\alpha_t(i)$, the MoEZip score is:

$$\phi_t(i) = \alpha_t(i) + \lambda \cdot \mathcal{F}_t(i)$$

where $\lambda$ is a hyperparameter balancing the two terms. Entries with the lowest $\phi_t(i)$ values are evicted.

---

## Learning or Inference Procedure

MoEZip is a training-free method applied at inference time:

1. During the prefill phase, compute $\mathcal{F}_t(i)$ and $\alpha_t(i)$ for each KV position and layer.
2. Sort positions by $\phi_t(i)$ in descending order.
3. Retain the top-$k$ positions (where $k$ controls the KV retention ratio) and evict the rest.
4. Proceed with autoregressive generation over the compressed KV cache.

No gradient computation or model fine-tuning is required. The method is compatible with any MoE LLM that uses token-level routing and is agnostic to the number of experts or layers.

---

## What the Paper Claims

- **Accuracy at high compression:** At 10% KV retention, MoEZip outperforms KVZip (the prior state-of-the-art) by +24 average points on LongBench and +20 points on RULER.
- **Multiplicative gains:** These represent 2.3× and 4.9× higher absolute scores relative to KVZip at the same retention ratio.
- **Generality:** Results are reported for both softmax-type and sigmoid-type MoE routing architectures, covering multiple deployed MoE LLMs.
- **Training-free:** The method adds negligible overhead at inference and requires no additional data or fine-tuning.

---

## Experimental Findings

- **Benchmarks:** LongBench (long-document QA, summarization, few-shot), NIAH (needle-in-a-haystack retrieval), and RULER (structured retrieval and reasoning).
- **Baselines:** H2O, SnapKV, PyramidKV, and KVZip across different MoE models.
- **Key results:** At high compression ratios (5–20% retention), MoEZip consistently outperforms all baselines. At higher retention ratios (≥50%), the gap narrows as all methods converge toward full-cache performance.
- **Both routing types:** Gains hold for both softmax-type MoE models (e.g., Mixtral-style) and sigmoid-type models (e.g., DeepSeek-style).

---

## Ablations and Interpretation

- **Routing-only scoring:** Using routing sensitivity alone without attention score degrades performance on tasks where certain instruction tokens have high attention but low routing impact, suggesting both signals are needed.
- **Attention-only scoring:** Without the Fisher term, performance matches existing dense-transformer methods—confirming that the routing-aware component drives the gains.
- **Sensitivity of λ:** Performance is robust to moderate changes in λ, with a mild sweet spot around λ = 1 for most benchmarks.

---

## Reference

Minsang Kim, Seungjun Baek. **MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs**. arXiv:2610.30206. NeurIPS 2026.  
https://arxiv.org/abs/2610.30206
