# MemSFT: Mitigating Alignment Tax with an External Parametric Memory

**Authors:** Jiarui Wang, Xiang Shi, Jiaqi Cao, Rubin Wei, Xiquan Wang, Hao Sun, Jingzhi Wang, Zhiqi Yang, Qipeng Guo, Bowen Zhou, Zhouhan Lin  
**Affiliation:** Shanghai Jiao Tong University; Fudan University; Tencent AI Lab  
**Venue:** arXiv:2607.25614  
**Date:** July 28, 2026  
**URL:** https://arxiv.org/abs/2607.25614

---

## Summary

Fine-tuning a general-purpose LLM on domain-specific data consistently improves performance in that domain but reliably degrades general capabilities—a phenomenon called the alignment tax or catastrophic forgetting. MemSFT proposes a structural alternative: instead of updating the backbone model's parameters, train a compact plug-and-play parametric memory module to encode domain knowledge. The memory imitates a non-parametric retriever operating over domain documents. During generation, a learned router fuses memory and backbone output distributions token-by-token. Because the backbone is untouched, general capability is preserved by construction. The memory is domain-specific but backbone-agnostic: once trained, it can be reused with LLMs of different sizes.

## Problem

The alignment tax is a practical obstacle to deploying LLMs in specialized domains (healthcare, law, science). Three approaches exist, each with limitations:

1. **Full SFT:** Updates all backbone parameters on domain data. Achieves strong domain performance but causes catastrophic forgetting—general benchmarks drop by 15–30% relative to the base model.
2. **LoRA/adapter fine-tuning:** Updates a small adapter. Reduces forgetting but does not eliminate it; domain gains are smaller than full SFT.
3. **Retrieval-augmented generation (RAG):** Retrieves relevant documents at inference. Preserves general capability but increases inference latency (retrieval over large corpora), requires a well-maintained document store, and doesn't internalize patterns.

MemSFT aims for the domain performance of full SFT with the general capability preservation of RAG, while eliminating the latency of retrieval.

## Method: External Parametric Memory

The key insight is that a non-parametric retriever operating over domain data implicitly represents the domain distribution: given a query prefix, it retrieves documents that shift the probability distribution toward domain-relevant continuations. MemSFT trains a parametric memory module $M_\phi$ to imitate this retrieval behavior:

1. **Memory architecture:** $M_\phi$ is a small transformer (50–100M parameters) with no cross-attention to the backbone. Given the current generation prefix $h_{1:t}$, it produces a distribution $M_\phi(\cdot | h_{1:t})$ over the next token.
2. **Memory training:** $M_\phi$ is trained to match the output distribution of a non-parametric retriever (e.g., a dense retriever + reranker + a frozen LM for completion scoring) operating over domain documents. This is a knowledge distillation objective:

$$\mathcal{L}_{\text{mem}} = D_{\text{KL}}\!\left(\text{Retriever}(\cdot | h_{1:t},\, \mathcal{D}_{\text{domain}}) \;\|\; M_\phi(\cdot | h_{1:t})\right)$$

3. **Router training:** A lightweight gating network $g_\psi(h_{1:t}) \in [0,1]$ is trained jointly to predict when domain memory should be invoked. The final token distribution is:

$$p(w | h_{1:t}) = g_\psi \cdot M_\phi(w | h_{1:t}) + (1 - g_\psi) \cdot p_{\text{backbone}}(w | h_{1:t})$$

## Technical Formulation

Let $p_\theta$ be the backbone LLM with frozen parameters $\theta$. Let $M_\phi$ be the memory module with trainable parameters $\phi$. Let $g_\psi$ be the router. The full generation distribution is:

$$p_{\text{total}}(w_t | w_{<t}) = g_\psi(w_{<t}) \cdot M_\phi(w_t | w_{<t}) + (1 - g_\psi(w_{<t})) \cdot p_\theta(w_t | w_{<t})$$

The router $g_\psi$ is trained to maximize the likelihood of held-out domain data under $p_{\text{total}}$:

$$\mathcal{L}_{\text{router}} = -\mathbb{E}_{(w, c) \sim \mathcal{D}_{\text{domain}}}\left[\log p_{\text{total}}(w | c)\right]$$

A sparsity regularizer $\lambda \|g_\psi\|_1$ is added to encourage the router to invoke memory selectively rather than always blending both distributions.

## Learning Procedure

Training proceeds in two phases:

1. **Memory distillation:** Train $M_\phi$ against the retriever's output distribution on domain documents. This requires no backbone fine-tuning. Training uses Adam for 10K steps on domain-specific corpora.
2. **Router training:** Freeze $M_\phi$ and train $g_\psi$ on a held-out set from the same domain. This is a small number of parameters (~1M) trained quickly (1K steps).

At inference, the backbone runs a single forward pass; the memory module runs in parallel and the router blends the outputs. The additional latency is approximately $15\%$ of the backbone's decoding latency for Qwen3-8B.

## What the Guarantee Says

Because the backbone parameters $\theta$ are never modified, the general-task performance of the backbone is preserved exactly. Any degradation in general tasks must come from the router incorrectly invoking domain memory in non-domain contexts. The sparsity regularizer bounds this: on non-domain prompts, the router converges to $g_\psi \approx 0$, reverting to pure backbone outputs.

## Experimental Findings

**Domains evaluated:** Biology (PubMedQA, BioASQ), Geoscience (GeoQA), Law (LegalBench).  
**Models:** Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-72B, Qwen3-235B-A22B.

| Method | Domain avg. (↑) | General avg. (↑) | Δ General |
|---|---|---|---|
| Base LLM | 62.3 | 74.8 | — |
| Full SFT | 71.8 | 54.3 | −20.5 |
| LoRA | 68.4 | 70.1 | −4.7 |
| RAG | 70.2 | 74.7 | −0.1 |
| MemSFT (ours) | **72.6** | **74.6** | **−0.2** |

MemSFT achieves 0.8 points higher domain performance than full SFT while degrading general performance by only 0.2 points—essentially matching RAG in general quality while exceeding RAG in domain performance by 2.4 points, and with 3× lower inference latency than RAG.

**Cross-model reuse:** A memory module trained with Qwen3-8B as teacher for the distillation step transfers effectively to Qwen3-72B, recovering 92% of the per-domain performance of a memory trained natively with the larger backbone.

## Ablations and Interpretation

**Router sparsity:** Removing the sparsity regularizer leads the router to always blend both distributions ($g_\psi \approx 0.5$), hurting general performance by an additional 3.1 points while providing only modest domain gains.

**Memory size:** Scaling memory from 25M to 100M parameters improves domain performance by 2.4 points with diminishing returns beyond 75M parameters.

**Distillation teacher quality:** Using a weaker retriever (BM25 vs. dense retrieval) reduces domain performance by 3.8 points, indicating that the quality of the knowledge distillation source is a primary bottleneck.
