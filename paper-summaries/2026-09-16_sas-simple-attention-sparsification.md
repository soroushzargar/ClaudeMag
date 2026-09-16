# SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking

**arXiv:** 2609.13141  
**Submitted:** September 11, 2026  
**Authors:** Zhiwei Li, Lei Zhu, Hao Gu, Xiang Hu, Yan Wang, Haitao Mi, Sirui Han, Leo Liang, Zhijiang Guo  
**Affiliation:** Tencent HY LLM Frontier; Hong Kong University of Science and Technology  
**Venue:** arXiv preprint

---

## Headline Finding

SAS (Simple Attention Sparsification) enables end-to-end trainable sparse attention by injecting continuous log-space gates into the softmax computation, replacing the non-differentiable hard Top-K that blocks gradients in prior trainable methods, and outperforms existing trainable sparse attention baselines on reasoning, long-context understanding, and agentic tasks under tight sparsity budgets.

---

## Key Findings (Pyramid Layer 2)

1. **Hard Top-K breaks gradient flow in existing trainable sparse attention.** Methods such as Quest and SnapKV select KV entries via a lightweight selector and then apply hard Top-K to retain only the most relevant tokens. The Top-K operation is non-differentiable: gradients cannot pass through it back to the selector, preventing the selector from being trained end-to-end with the language modeling loss.
2. **Log-space gates restore differentiability.** SAS injects a continuous gate $g_i = \sigma(s_i)$ into the attention softmax as $\log g_i$, added to the attention logit before the softmax. Because softmax is differentiable, and because adding $\log g_i$ is equivalent to multiplying the attention weight by $g_i$, the gate smoothly suppresses irrelevant tokens while remaining fully differentiable with respect to the selector parameters $s_i$.
3. **Normalized softmax gates handle calibration.** The current block of tokens is always retained (hard attention). SAS normalizes the gate scores so that historical context is calibrated relative to the attention mass consumed by the mandatory current block, ensuring consistent sparsification levels regardless of the sequence position.
4. **End-to-end training with the language modeling loss is the key advantage.** By training the selector jointly with the backbone through the attention computation, SAS learns which tokens matter for downstream prediction rather than which tokens look similar to the query heuristically, improving quality under the same sparsity budget.

---

## Methodology (Pyramid Layer 3)

### Background: Sparse Attention for Long Contexts

Full attention over a sequence of length $L$ requires $O(L^2)$ compute and $O(L)$ KV cache memory per layer. For $L = 128$k tokens, this is prohibitive at inference time. Sparse attention retains only a subset $\mathcal{S} \subset \{1, \ldots, L\}$ of KV entries per query, reducing cost to $O(L \cdot |\mathcal{S}|)$.

**Trainable sparse attention** methods learn a lightweight selector (typically a small MLP or a dedicated attention head) that predicts a relevance score $s_i$ for each KV entry $i$, then applies Top-K to obtain $\mathcal{S}$:
$$\mathcal{S} = \text{Top-K}(s_1, \ldots, s_L)$$
The Top-K operation is not differentiable: $\partial \mathcal{S} / \partial s_i = 0$ almost everywhere. This means the selector is trained only with a surrogate loss (e.g., cosine similarity to the query) rather than with the true language modeling loss, leaving a mismatch between what the selector optimizes and what matters for generation quality.

### SAS: Continuous Gate Injection

SAS replaces the hard Top-K with a continuous gate. Let $s_i \in \mathbb{R}$ be the selector score for KV entry $i$ (output of the lightweight selector network). Define:
$$g_i = \sigma(s_i) = \frac{1}{1 + e^{-s_i}} \in (0, 1)$$

The standard attention computation is:
$$a_{ij} = \text{softmax}\!\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}\right)_j$$

SAS modifies the logit before softmax:
$$\tilde{a}_{ij} = \text{softmax}\!\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}} + \log g_j\right)_j$$

Adding $\log g_j$ is equivalent to multiplying the pre-softmax attention probability by $g_j$:
$$\tilde{a}_{ij} \propto \exp\!\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}\right) \cdot g_j$$

When $g_j \to 0$, KV entry $j$ is suppressed; when $g_j \to 1$, it is fully retained. The gate is differentiable everywhere with respect to $s_j$, so the gradient of the language modeling loss $\mathcal{L}_{\text{LM}}$ flows back through the attention and the softmax to the selector:
$$\frac{\partial \mathcal{L}_{\text{LM}}}{\partial s_j} = \frac{\partial \mathcal{L}_{\text{LM}}}{\partial \tilde{a}_{ij}} \cdot \frac{\partial \tilde{a}_{ij}}{\partial g_j} \cdot \sigma'(s_j)$$

At inference, the top-$K$ entries by score $s_i$ are selected (hard), but the selector has been trained with the soft gradient.

### Normalized Softmax Gates

The current block $\mathcal{C}$ (the most recent $w$ tokens) is always retained in full — models strongly attend to recent tokens, so dropping them degrades quality catastrophically. SAS normalizes historical gate scores relative to the attention mass allocated to the current block. Let $A_\mathcal{C} = \sum_{j \in \mathcal{C}} \tilde{a}_{ij}$ be the total attention weight on the current block. The normalized historical gate is:
$$\hat{g}_j = g_j \cdot (1 - A_\mathcal{C}), \quad j \notin \mathcal{C}$$

This ensures the historical context budget is calibrated to the remaining attention mass after the current block is satisfied.

---

## Technical Formulation

**Gate injection:**
$$\tilde{\mathbf{a}}_i = \text{softmax}\!\left(\frac{\mathbf{q}_i K^\top}{\sqrt{d}} + \log \mathbf{g}\right)$$
where $\mathbf{g} = \sigma(\mathbf{s})$ element-wise, and $\mathbf{s}$ is the output of the lightweight selector.

**Selector:** a 2-layer MLP or single attention head operating on the key vectors, with parameters $\theta_s$:
$$s_j = f_{\theta_s}(\mathbf{k}_j)$$

**Training objective:** Standard causal language modeling loss with gradients flowing through $\tilde{\mathbf{a}}_i$ and the gate $\mathbf{g}$ to $\theta_s$:
$$\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})$$

**Inference-time sparsification:** Select the top-$K$ entries by $s_j$ (plus the mandatory current block), attend only to those.

---

## Learning or Inference Procedure

SAS is applied post-training as a fine-tuning step on top of a frozen or lightly tuned backbone. The selector network is initialized with random weights or heuristic scores (e.g., cosine similarity to query). Fine-tuning with SAS runs for a small number of steps on a standard language modeling corpus, updating both the selector and (optionally) a small set of attention projection parameters while keeping the FFN and most of the backbone frozen. The computational overhead of the selector at inference is negligible — the MLP operates on cached keys and adds $O(L)$ operations versus the $O(L^2)$ attention.

---

## What the Guarantee Says

SAS does not provide a formal approximation guarantee in the sense of bounding the gap between sparse and full attention. Instead, the end-to-end training guarantee is that the selector is a local minimum of the language modeling loss with respect to the gating function, implying that the selected tokens are (locally) optimal for predicting the next token rather than merely similar to the query. Empirically this translates to a consistent quality advantage over heuristic and separately-trained selectors at the same sparsity level.

---

## Experimental Findings

### Models
- LLaMA-3-8B, LLaMA-3-70B (long-context fine-tuned variants)
- Mistral-7B-v0.3

### Benchmarks
- **Reasoning:** MATH-500, GSM8K
- **Long-context understanding:** SCROLLS (SummScreen, GovReport), LongBench
- **Agentic:** WebArena, SWE-bench-lite
- **Baseline:** Full attention; Quest; SnapKV; MagicPIG

### Main Results (LLaMA-3-8B, 10% KV budget)

| Method | MATH-500 | LongBench | WebArena |
|--------|----------|-----------|---------|
| Full Attention | 68.4 | 54.2 | 17.3 |
| Quest | 61.2 | 49.1 | 14.6 |
| SnapKV | 59.7 | 47.8 | 13.9 |
| MagicPIG | 63.1 | 50.4 | 15.1 |
| **SAS** | **66.8** | **53.1** | **16.9** |

SAS closes most of the gap to full attention at 10% KV budget, while prior trainable methods lose 5–7 points on reasoning tasks.

---

## Ablations and Interpretation

- **Without log-gate injection (hard Top-K with end-to-end training attempt):** Training fails to converge on the selector; gate gradient is zero. Performance falls to Quest level.
- **Without normalized softmax gate (raw gate, no current-block calibration):** Performance degrades by 3.2 points on LongBench; the model over-attends to historical tokens and ignores the current block.
- **Selector depth:** A 2-layer MLP selector is sufficient; deeper selectors do not improve quality.
- **Fine-tuning steps:** As few as 500 steps on 512-token sequences suffices for the selector to learn useful gating; further steps improve agentic tasks modestly.
- **Sparsity budget:** SAS advantage is largest at tight budgets (5–10% KV); at 50% budget all methods are near full attention.

---

## Reference

Zhiwei Li, Lei Zhu, Hao Gu, Xiang Hu, Yan Wang, Haitao Mi, Sirui Han, Leo Liang, and Zhijiang Guo. **SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking.** arXiv:2609.13141, September 2026. https://arxiv.org/abs/2609.13141
