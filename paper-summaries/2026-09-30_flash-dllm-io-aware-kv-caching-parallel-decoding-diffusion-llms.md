# Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs

**arXiv:** 2609.26796  
**Authors:** Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  
**Affiliation:** VILA Lab, Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)  
**Date:** September 22, 2026

---

## One-Line Summary

Flash-dLLM identifies GPU memory I/O — not compute — as the dominant bottleneck in diffusion LLM inference with KV caching, and addresses it with a fused IO-aware kernel plus a self-drafting decode strategy that yields 5.1× and 11.0× speedups over the strongest baseline on GSM8K and HumanEval.

---

## Problem

Diffusion language models (dLLMs) are a promising non-autoregressive alternative to standard autoregressive LLMs: they generate entire sequences by iteratively denoising masked tokens rather than decoding left-to-right. This enables parallel token generation. However, the practical deployment of dLLMs is hampered by slow inference, and recent efforts to apply KV caching to dLLMs (analogous to how it accelerates autoregressive models) have underperformed expectations. The core bottleneck — GPU memory I/O rather than floating-point operations — has not been systematically diagnosed.

---

## Why Existing Approaches Fall Short

- **Standard autoregressive KV caching** does not directly transfer to dLLMs because the full sequence is present at every denoising step; the access pattern is bidirectional and the cache must be read in full at each iteration.
- **Elastic-Cache and related dLLM accelerators** apply token eviction or cache compression but do not address the IO bottleneck: the remaining cache still requires many memory reads per denoising step.
- **Parallel decoding proposals for dLLMs** rely on an external draft model to propose token sequences that the dLLM verifies, adding inference-time model size and latency.
- **Flash Attention** for autoregressive models optimizes compute kernels but was not designed for the bidirectional, full-sequence read pattern of dLLMs.

---

## Core Method

**IO-Aware Fused KV-Cache Kernel.** Flash-dLLM profiles the dLLM forward pass and identifies GPU HBM reads as the dominant bottleneck when KV caching is applied. The paper introduces a fused CUDA kernel that combines the KV-cache read, attention computation, and output write into a single kernel launch. By keeping intermediate values in on-chip SRAM and eliminating redundant round-trips to HBM, the kernel reduces memory bandwidth consumption at each denoising step.

**Self-Drafting Decode Strategy.** Instead of requiring an auxiliary draft model, Flash-dLLM proposes using the dLLM itself as both drafter and verifier. Early denoising steps (low noise level) produce confident token predictions; these predictions serve as a draft that later, more refined steps verify and correct. The KV cache from early steps is reused by later steps, amortizing the IO cost over multiple decoding iterations.

---

## Technical Formulation

Let $\mathbf{x}^{(t)}$ denote the partially denoised sequence at denoising step $t$, and let $f_\theta$ be the dLLM transformer. The standard denoising update:
$$\mathbf{x}^{(t-1)} = f_\theta\!\left(\mathbf{x}^{(t)}, t\right)$$
requires reading the full KV cache at each step $t$. For a sequence of length $L$ with $d$-dimensional keys and values over $H$ heads and $l$ layers, the IO cost per step is:
$$\text{IO}(t) = 2 \cdot L \cdot d \cdot H \cdot l \cdot \text{sizeof}(\text{float16})$$

Flash-dLLM's fused kernel computes tiled attention on-chip:
$$\text{IO}_{\text{fused}}(t) = \frac{L \cdot d \cdot H \cdot l}{B_{\text{SRAM}}} \cdot \text{sizeof}(\text{float16})$$
where $B_{\text{SRAM}}$ is the SRAM tile size, reducing HBM reads by a factor proportional to tile utilization.

**Draft-and-Verify with Self-Reuse.** At step $t$, the model generates a draft $\hat{\mathbf{x}}^{(0)}$ from the cached KV of step $t$. At step $t-1$, the KV cache from step $t$ is partially reused: positions where $\hat{\mathbf{x}}^{(0)}$ matches the final prediction at step $t-1$ avoid recomputation.

---

## Learning or Inference Procedure

Flash-dLLM is a **training-free, inference-time** system:

1. **Profile:** Identify the IO-bound attention layers in the target dLLM.
2. **Build fused kernel:** Compile the IO-aware CUDA kernel for the target GPU architecture (A10G, A100, etc.).
3. **Self-drafting loop:**  
   - Run the dLLM's first $T_{\text{draft}}$ denoising steps with the fused kernel, producing draft token predictions.  
   - Verify and correct drafts in the remaining $T - T_{\text{draft}}$ steps, reusing as much KV cache as possible.
4. **Output:** The final denoised sequence at step 0.

No model weights are modified; the method works on any transformer-based dLLM.

---

## What the Guarantee Says

Flash-dLLM guarantees exact equivalence of output to the unoptimized dLLM (the fused kernel computes the same numerical result as the unfused baseline up to floating-point rounding). Speedup comes entirely from reducing IO operations, not from approximation. The self-drafting strategy introduces a soft optimism bias — it assumes later steps will agree with earlier ones — but the verification step ensures correction.

---

## Experimental Findings

Evaluated against Elastic-Cache (the strongest prior dLLM acceleration method) on:

- **GSM8K** (mathematical reasoning): **5.1× speedup** over Elastic-Cache.
- **HumanEval** (code generation): **11.0× speedup** over Elastic-Cache.

The code benchmark shows a larger speedup because code generation involves longer sequences with sparser denoising trajectories, making IO savings more impactful.

Additional results demonstrate improved memory efficiency (lower peak HBM usage) at equal or better generation quality, confirming that the fused kernel reduces memory footprint as well as latency.

---

## Ablations and Interpretation

- **Fused kernel vs. naive kernel:** The fused kernel alone accounts for 3–5× of the speedup; the draft-and-verify strategy contributes the remaining 1.5–2×.
- **Draft length $T_{\text{draft}}$:** Shorter drafts reduce reuse and speedup; longer drafts risk more corrections but still improve throughput because IO savings outweigh correction costs.
- **IO profile analysis:** On all tested GPUs (A10G, A100), attention accounts for more than 70% of total inference time when KV caching is applied, confirming the IO bottleneck hypothesis.
- **Comparison to autoregressive models:** dLLMs with Flash-dLLM match or approach autoregressive LLMs with Flash Attention in wall-clock inference speed for the same model size.

---

## Reference

Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen. **Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs**. arXiv:2609.26796, September 2026.  
https://arxiv.org/abs/2609.26796
