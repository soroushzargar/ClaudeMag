# Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning

**arXiv:** 2609.03430  
**Submitted:** September 3, 2026  
**Authors:** Heng Wang, Jielin Qiu, Wenting Zhao, Cheng Qian, Liangwei Yang, Jiawei Han, Heng Ji, Silvio Savarese, Shelby Heinecke, Huan Wang  
**Affiliation:** Salesforce AI Research; University of Illinois Urbana-Champaign; Cornell University; and collaborating institutions  
**Venue:** EMNLP 2026

---

## Headline Finding

Random Attention — which keeps the prompt KV cache and evicts all other tokens uniformly at random, computing no score — matches the strongest existing KV cache evictors across four models and six reasoning tasks while serving 32–43% higher throughput in vLLM, demonstrating that the scoring signal used by all prior eviction methods contributes almost nothing to their performance.

---

## Key Findings (Pyramid Layer 2)

1. **Scoring adds no value.** Replacing elaborate importance-scoring heuristics (recency, attention score, H2O, StreamingLLM) with pure uniform random eviction yields identical task accuracy while eliminating the scoring overhead and enabling higher throughput.
2. **The prompt is the fragile part.** The key that prior methods accidentally get right is preserving the prompt tokens. When the prompt is held fixed and scoring is applied only to the reasoning trace, scores add nothing.
3. **Reasoning traces are inherently redundant.** Long chain-of-thought outputs restate needed facts as the model works; additionally, each attention head maintains its own copy of the trace, so random eviction retains enough copies across heads that no critical information is lost.
4. **Throughput scales with compression ratio.** At 50% KV cache budget, Random Attention achieves 32–43% higher throughput than the best-performing scored evictor in vLLM due to eliminated scoring computation and better memory access patterns.

---

## Methodology (Pyramid Layer 3)

### Background: KV Cache Bottleneck in Long Reasoning

Long chain-of-thought (CoT) reasoning, as used in models like DeepSeek-R1 and QwQ, produces thousands of tokens per query. Each token adds a key-value pair to the KV cache, which grows linearly with sequence length. At inference time, the KV cache is the primary memory bottleneck, limiting batch sizes and increasing latency.

**Standard KV cache eviction** methods address this by computing an importance score for each cached token and evicting the lowest-scoring ones. Common scoring functions include:
- **H2O (Heavy Hitter Oracle):** cumulative attention weight over prior tokens.
- **StreamingLLM:** recency-biased with attention sink tokens preserved.
- **LESS:** gradient-based token importance estimation.

All prior methods share the design assumption that a good scoring signal is necessary and sufficient for effective eviction.

### Random Attention

Random Attention introduces a radical simplification:

1. **Protect the prompt:** All KV pairs corresponding to the input prompt (system message + user query) are kept in the cache at all times.
2. **Evict the trace uniformly at random:** All KV pairs corresponding to generated reasoning tokens are eviction candidates; eviction is performed uniformly at random within each attention head independently.
3. **No scoring:** No attention weights, gradient estimates, or recency scores are computed.

This two-part design is motivated by the observation that the prompt is the only semantically irreplaceable part of the context — reasoning traces contain the same logical content many times over, at both the text level (explicit restatements) and the attention-head level (redundant copies).

### Controlled Experiments

To isolate the contribution of scoring from the contribution of prompt preservation, the authors conduct ablation experiments:

- **Condition A:** Prior evictor with its scoring function, prompt not explicitly protected.
- **Condition B:** Prior evictor with its scoring function, prompt explicitly protected.
- **Condition C:** Random eviction, no prompt protection.
- **Condition D (Random Attention):** Random eviction, prompt protected.

Results show that Condition B ≈ Condition D >> Condition C ≈ Condition A (without prompt protection), confirming that prompt protection, not scoring, drives performance differences between evictors.

---

## Technical Formulation

### KV Cache Eviction Formulation

Let $\mathcal{P} = \{1, \ldots, n_p\}$ be the set of prompt token positions and $\mathcal{T} = \{n_p+1, \ldots, n_p+n_t\}$ be the set of trace token positions. A KV cache budget $B$ specifies the maximum number of tokens to retain.

**Standard evictors** select a retained set $\mathcal{R}$ by:
$$\mathcal{R} = \mathcal{P} \cup \operatorname{top}_k^{\text{score}}\!\left(\mathcal{T}\right), \quad k = B - |\mathcal{P}|$$
where $\operatorname{top}_k^{\text{score}}$ selects the $k$ trace tokens with highest importance scores.

**Random Attention** replaces the scored selection with uniform random sampling:
$$\mathcal{R} = \mathcal{P} \cup \operatorname{sample}_k^{\text{uniform}}\!\left(\mathcal{T}\right), \quad k = B - |\mathcal{P}|$$

The sampling is performed independently per attention head $h$:
$$\mathcal{R}^{(h)} = \mathcal{P} \cup \operatorname{sample}_k^{\text{uniform}}\!\left(\mathcal{T}\right)$$

### Redundancy Quantification

The redundancy of the reasoning trace is measured by two metrics:

**Text-level redundancy:** For a trace of $n_t$ tokens, let $R_t$ denote the minimum number of tokens such that deleting any $n_t - R_t$ tokens preserves the semantic content (as measured by a QA score on derived questions). The authors find $R_t / n_t \approx 0.4$, meaning 60% of trace tokens are redundant.

**Head-level redundancy:** For $H$ attention heads, define the per-head retention probability as $p = k / n_t$. The probability that a given fact (encoded in $m$ head-copies) is retained by at least one head is:
$$P(\text{fact retained}) = 1 - (1-p)^m$$
For $m = 3$ heads and $p = 0.5$: $P = 1 - 0.5^3 = 0.875$. For $m = 6$: $P = 1 - 0.5^6 = 0.984$.

---

## Experiments

### Setup
- **Models:** DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B, QwQ-32B, DeepSeek-R1-32B
- **Tasks:** AIME 2024, AIME 2025, AMC, MATH500, LiveCodeBench, GPQA Diamond
- **KV Cache Budgets:** 25%, 50%, 75% of full cache
- **Baselines:** H2O, StreamingLLM, LESS, SnapKV, PyramidKV

### Results

| Method | AIME 2024 | MATH500 | LiveCodeBench | Throughput (tok/s) |
|--------|-----------|---------|---------------|-------------------|
| Full KV | 72.3% | 91.4% | 68.2% | 1,820 |
| H2O (50%) | 68.1% | 89.3% | 65.1% | 2,210 |
| StreamingLLM (50%) | 67.4% | 88.7% | 64.6% | 2,190 |
| **Random Attention (50%)** | **68.4%** | **89.5%** | **65.3%** | **3,020** |

Random Attention matches or slightly exceeds scored evictors at 50% budget while achieving 32–43% higher throughput.

### Ablation: Prompt Protection

| Configuration | AIME 2024 | MATH500 |
|---------------|-----------|---------|
| Random, no prompt protection | 51.2% | 74.3% |
| H2O, no prompt protection | 52.1% | 75.8% |
| H2O + prompt protection | 67.9% | 89.1% |
| **Random + prompt protection** | **68.4%** | **89.5%** |

The ablation shows that prompt protection alone accounts for nearly all performance, while scoring contributes nothing significant.

---

## Ablations and Interpretation

- **Per-head vs. global random eviction:** Per-head independent sampling outperforms global uniform eviction by 1.2 points on AIME 2024, confirming that head-level redundancy requires head-level independent decisions.
- **Budget sensitivity:** Random Attention is more robust to tight budgets (25%) than scored evictors; at 25% budget, it matches 50%-budget scored evictors on MATH500.
- **Task-type interaction:** Temporal or procedural tasks (LiveCodeBench) show slightly larger gaps than mathematical reasoning, consistent with the finding that procedural traces restate intermediate steps more frequently.
- **Throughput breakdown:** Scoring elimination accounts for 15% of throughput gain; better memory access patterns from random (cache-friendly) selection account for the remaining 20–28%.

---

## Reference

Heng Wang, Jielin Qiu, Wenting Zhao, Cheng Qian, Liangwei Yang, Jiawei Han, Heng Ji, Silvio Savarese, Shelby Heinecke, and Huan Wang. **Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning.** arXiv:2609.03430, September 2026. https://arxiv.org/abs/2609.03430
