# Mamba with Hierarchical Memory: Solving Representation Bottleneck in Long Sequence Modeling

**arXiv:** 2608.02347
**Authors:** Qinwen Wang, Jieping Luo, Aoxiang Qin, Ruoyu Zhao, Jianxiong Tang,
             Wei Zhang, Zhichao Lu, Luziwei Leng
**Venue:** arXiv preprint (August 2026)
**Date:** August 3, 2026
**URL:** https://arxiv.org/abs/2608.02347

---

## Summary

Hierarchical Memory Mamba (HMM) addresses the fundamental representation bottleneck that limits recurrent linear attention models like Mamba on long sequences. Drawing on the cognitive science of hierarchical human memory—sensory, working, and long-term memory—HMM adds a lightweight working memory module that compresses paragraph-level semantics from Mamba's fast hidden states and stores them in a persistent long-term memory for task-relevant retrieval. This allows the model to maintain relevant context across arbitrarily long sequences without expanding the fixed recurrent state.

## Problem: The Recurrent Representation Bottleneck

Mamba and related State Space Models (SSMs) process sequences in linear time $O(N)$ without an explicit KV cache, making them highly efficient. However, this efficiency comes at a cost: **the recurrent hidden state has fixed capacity**. At each step, the hidden state must compress all prior context into a fixed-size vector. As sequence length grows, older information is progressively overwritten, creating what the authors call the **representation bottleneck**.

In practice this manifests as degraded performance on:
- Long-document QA (questions depend on early context)
- Many-shot in-context learning (earlier examples are forgotten)
- Summarization of book-length texts
- Long-horizon reasoning chains

Transformers avoid this bottleneck with their KV cache, but at $O(N^2)$ attention cost. HMM aims to give SSMs long-range memory without sacrificing linear time complexity.

## Hierarchical Memory Design

Inspired by the three-tier model of human memory, HMM adds two memory tiers above Mamba's base recurrent state:

**Tier 1 — Sensory Memory (Mamba's hidden state $h_t$):** Fast, high-capacity, short-lived. The standard Mamba recurrent state processes the current token in context of recent tokens. This tier handles fine-grained local dependencies but degrades on long-range dependencies.

**Tier 2 — Working Memory (PLS extractor):** A lightweight module that periodically reads Mamba's hidden states over a paragraph-length window and extracts **Paragraph-Level Semantics (PLS)**—a compact summary of the paragraph's key information. PLS extraction uses a small cross-attention module:
$$\mathrm{PLS}_k = \mathrm{CrossAttn}(Q_{\mathrm{probe}},\; K = h_{[k\cdot P : (k+1)\cdot P]},\; V = h_{[k\cdot P : (k+1)\cdot P]})$$
where $P$ is the paragraph length and $Q_{\mathrm{probe}}$ is a learned query vector.

**Tier 3 — Long-Term Memory (LTM store):** PLS vectors are written to a persistent memory bank that grows with sequence length. At inference, the model retrieves relevant entries via a learned key-value attention:
$$\mathrm{retrieved}_t = \mathrm{Attn}(h_t,\; K_{\mathrm{LTM}},\; V_{\mathrm{LTM}})$$
Retrieved content is injected into Mamba's processing via a gating mechanism, allowing long-range context to influence current token predictions without storing it in the recurrent state.

## Complexity Analysis

- **PLS extraction:** Runs once per $P$ tokens, $O(N/P)$ write operations
- **LTM retrieval:** $O(N/P)$ memory entries, retrieval cost $O(N/P)$ per step → total $O(N^2/P)$
- For practical paragraph sizes $P \gg 1$, the total cost approaches $O(N)$ for long-range dependencies, vs. $O(N^2)$ for full attention

This makes HMM asymptotically more efficient than Transformers for long sequences while retaining linear-time base processing.

## Results

HMM outperforms Mamba-2 and other SSM baselines on:
- **SCROLLS long-document benchmarks** (QuALITY, SummScreenFD, GovReport): large gains on tasks requiring cross-document reasoning
- **Long-range Arena**: improved scores across all six tasks, with the largest gains on tasks requiring long-range recall (Path-X, Retrieval)
- **Many-shot ICL**: consistent accuracy improvement as the number of in-context examples grows (Mamba saturates and then degrades; HMM continues to improve)
- **Perplexity on long-context language modeling** (PG-19, Books3): lower perplexity than Mamba-2 on sequences exceeding 8k tokens

## Why It Matters

SSMs are attractive for long-sequence applications (genomics, audio, long documents) precisely because of their linear complexity. But the representation bottleneck has limited their practical long-context performance below Transformer+KV-cache baselines. HMM provides a lightweight, interpretable solution grounded in cognitive memory science, adding only a small fraction of the base model's parameters. The hierarchical structure is modular and can be bolted onto existing trained Mamba checkpoints with minimal fine-tuning.

## Key Contributions

1. Formal characterization of the representation bottleneck in fixed-state RLAs
2. Hierarchical Memory Mamba: PLS extractor (working memory) + LTM store
3. Near-linear complexity for long-range dependencies
4. Strong empirical results on long-document QA, many-shot ICL, and long-range arena
