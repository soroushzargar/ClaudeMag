# Progressive Cramming: Reliable Token Compression and What It Reveals

**Authors:** Dmitrii Tarasov, Timofei Lashukov, Elizaveta Goncharova, Andrey Kuznetsov  
**Venue:** arXiv:2607.21231  
**Date:** July 23, 2026  
**URL:** https://arxiv.org/abs/2607.21231

---

## Summary

Token cramming compresses long input sequences into a small set of learned continuous embeddings, enabling LLMs to reference long contexts through a small, fixed memory footprint. Existing work achieves near-perfect reconstruction (99%+ token-level accuracy) but uses a fixed budget and a 99% threshold, leaving a key question open: **are residual errors an optimization failure or a fundamental compression limit?**

This paper introduces **progressive cramming**, which resolves this ambiguity by growing the target prefix token-by-token and stopping only when reconstruction fails, revealing the *true* per-problem compression horizon. The key finding: even perfect reconstruction is insufficient for task performance—crammed embeddings interact destructively with the model's early attention layers in ways that cannot be fully repaired.

## Problem

Standard token cramming fixes a compression ratio (e.g., compress N tokens into M embeddings) and optimizes until a reconstruction accuracy threshold is reached. This methodology conflates two distinct failure modes:
1. **Optimization failure:** The optimizer did not converge to a crammed embedding that would, in principle, recover the prefix.
2. **Capacity limit:** No embedding vector of the given dimensionality can encode the prefix faithfully enough for the model to use it.

Distinguishing these is crucial for understanding what compression can and cannot achieve.

## Method: Progressive Cramming

Progressive cramming proceeds as follows:
1. Start with a compression target of 1 token (trivially easy).
2. Grow the target by 1 token per iteration.
3. At each step, run optimization to find the best crammed embedding for the current prefix length.
4. Stop when optimization fails to achieve a target reconstruction fidelity within a fixed FLOP budget.

This yields a **per-input progressive trajectory**: a sequence of crammed embeddings of increasing length, with each step optimally fit to the data. The maximum achievable length before failure defines the compression horizon for that input.

Key finding: **progressive trajectories lie in a low-dimensional manifold** in embedding space. The trajectory dimension is far less than the ambient embedding dimension, suggesting a structured latent code underlies successful compression.

## Technical Formulation

Let $x = (x_1, \ldots, x_N)$ be an input token sequence and $e \in \mathbb{R}^d$ be a crammed embedding. Reconstruction loss is:

$$\mathcal{L}_\text{recon}(e, x) = \sum_{i=1}^N \ell_\text{CE}\!\left(f_\theta(e, x_{<i}), x_i\right)$$

where $f_\theta$ is the frozen LLM. Standard cramming solves:

$$e^* = \argmin_e \mathcal{L}_\text{recon}(e, x_{1:N})$$

Progressive cramming instead solves a sequence of problems:

$$e^*_n = \argmin_e \mathcal{L}_\text{recon}(e, x_{1:n}), \quad n = 1, 2, \ldots$$

stopping at $n^*$ where $\mathcal{L}_\text{recon}(e^*_{n^*}, x_{1:n^*}) > \tau$ for a fixed budget $B$. The key metric is the **compression horizon** $n^*/N$: the fraction of the prefix compressible within the fixed budget.

## Intervention: Causal Attention Knockout

To diagnose *why* crammed embeddings hurt downstream task accuracy, the authors apply causal attention-knockout interventions: they ablate attention edges between the crammed embedding and each subsequent layer's attention heads independently, measuring the effect on final task accuracy. Results point consistently to **early-layer interactions** as the source of degradation—specifically, layers 1–4 in tested models. The crammed embedding interferes with normal position-dependent attention patterns that the model uses to orient itself within the context.

## Key Results

- **Moderate benchmark drop (multiple-choice):** Prepending a crammed embedding alongside the original prefix reduces accuracy by 4–9% on MMLU-style benchmarks—despite the model having access to the original prefix in full.
- **Near-total collapse (generative tasks):** Under open-ended generation, prepending a crammed embedding collapses capability almost entirely (>30% absolute drop in correctness metrics).
- **Compression horizon varies:** For 512-token prefixes, compression horizons range from 140 to 380 tokens depending on the input type, suggesting content-dependence in compression difficulty.
- **Low-dimensional manifold:** PCA on progressive trajectories reveals that 80% of variance is explained by the first 5 principal components, far fewer than the full embedding dimension (4096 in tested models).

## Significance

Progressive cramming reveals a **fundamental, content-dependent limit** to token compression that cannot be overcome by better optimization. Any deployment of token cramming for long-context compression must account for early-layer interference, and systems claiming near-perfect reconstruction should be tested at inference time rather than solely at reconstruction time.
