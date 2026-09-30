# Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference

**arXiv:** 2609.26621  
**Authors:** Gaoyuan Du, Anam Nawaz Khan, Rex Zhou, Xiaoyang Liu, Deepayan Chakrabarti, Fnu Suya, Xueping Li  
**Affiliation:** University of Tennessee Knoxville; University of Chicago; Amazon; University of Texas at Austin  
**Date:** September 2026

---

## One-Line Summary

Greedy decoding from the same model and prompt diverges on identical hardware in BF16 vs. FP16 for 49–100% of prompts across six models and three benchmarks; the divergence is not caused by deep-layer error accumulation but by the top-two logit margin at the language model head, and selective FP32 head recomputation recovers +22–36 percentage points of exact agreement at only ±4% latency overhead.

---

## Problem

Greedy decoding is the standard deterministic decoding strategy for LLMs: at each step, select the token with the highest log-probability. Because the model, hardware, and decoding algorithm are fixed, the output is assumed to be identical across runs. In practice, it is well known that small numerical differences arise from parallelism and accumulation order, but these are assumed to have negligible effect on outputs because the top-token probability margin is usually large. This paper challenges that assumption by comparing BF16 and FP16 inference — two widely deployed precision formats — on identical hardware and finds pervasive output divergence.

---

## Why Existing Approaches Fall Short

- **Assumed determinism:** Practitioners and evaluation frameworks routinely assume greedy outputs are precision-invariant, making cross-precision comparisons without this caveat.
- **Nondeterminism literature** (e.g., NeurIPS 2025 Oral) focused on same-precision variability from parallelism order and seeding, not cross-precision systematic divergence.
- **FP16 vs. BF16 comparison:** BF16 has a wider dynamic range but fewer mantissa bits; FP16 has more mantissa bits but narrower range. Neither is a strict superset of the other, leading to divergent rounding at individual operations.
- **Mitigation approaches** such as full FP32 inference are impractical at scale due to 2× memory and bandwidth cost. No cheap, targeted fix existed before this work.

---

## Core Method

**Systematic Characterization.** The paper evaluates six LLMs (1.1B–7B parameters, four model families) using greedy decoding in BF16 and FP16 on three benchmarks (MMLU, TriviaQA, GSM8K) on identical A10G hardware. For each prompt, it records whether the full output trajectory diverges (a single token flip or more).

**Empirical Error-Propagation Analysis.** For each diverging prompt, the paper traces where the first diverging token flip occurs in the transformer forward pass. It records the accumulated numerical error at each layer and at the language model (LM) head, and measures the top-two logit margin at the flip step.

**Predictive Model of Divergence.** The analysis finds that divergence is almost entirely determined by two quantities at the LM head, not by deep-layer error accumulation:
1. The top-two logit margin (the gap between the highest and second-highest logit).
2. The directional perturbation between BF16 and FP16 in the top-two logit directions.

**Selective FP32 Head Recomputation.** When the top-two margin is below a threshold $\tau$, the LM head computation is re-run at FP32 for that token step. Otherwise, the BF16/FP16 result is kept. This adds overhead only on the small fraction of steps where the margin is tight.

---

## Technical Formulation

Let $\mathbf{h} \in \mathbb{R}^d$ be the last-layer hidden state and $\mathbf{W} \in \mathbb{R}^{V \times d}$ the LM head weight matrix. Logits are $\mathbf{z} = \mathbf{W}\mathbf{h}$.

Under BF16 and FP16, the computed logit vectors are:
$$\mathbf{z}^{\text{bf16}} = \mathbf{W}^{\text{bf16}} \mathbf{h}^{\text{bf16}} + \boldsymbol{\epsilon}^{\text{bf16}}, \qquad
\mathbf{z}^{\text{fp16}} = \mathbf{W}^{\text{fp16}} \mathbf{h}^{\text{fp16}} + \boldsymbol{\epsilon}^{\text{fp16}}$$
where $\boldsymbol{\epsilon}$ is the accumulated floating-point rounding error.

A flip occurs when:
$$\arg\max \mathbf{z}^{\text{bf16}} \neq \arg\max \mathbf{z}^{\text{fp16}}$$

The paper shows empirically that:
$$\Pr[\text{flip}] \approx \Pr\!\left[\Delta_{\text{top2}} < \left\|\boldsymbol{\epsilon}^{\text{bf16}} - \boldsymbol{\epsilon}^{\text{fp16}}\right\|_{\text{top2-dir}}\right]$$
where $\Delta_{\text{top2}} = z_{(1)} - z_{(2)}$ is the top-two margin and $\|\cdot\|_{\text{top2-dir}}$ is the error norm projected onto the top-two candidate direction.

**Selective FP32 criterion:**
$$\text{recompute at FP32 iff} \quad \Delta_{\text{top2}} < \tau$$
where $\tau$ is tuned per hardware to balance coverage and overhead.

---

## Learning or Inference Procedure

No training is required. The mitigation is applied at inference time:

1. Run the forward pass in BF16 (or FP16) as usual.
2. At the LM head, compute logits and measure $\Delta_{\text{top2}}$.
3. If $\Delta_{\text{top2}} < \tau$: recompute the LM head matmul in FP32 and use the FP32 logits.
4. Select the greedy token from the (possibly recomputed) logits.
5. Proceed to the next step.

The threshold $\tau$ can be calibrated on a small validation set or set to a fixed percentile of observed margins.

---

## What the Guarantee Says

Selective FP32 recomputation is not a worst-case guarantee — the approach cannot guarantee identical outputs to any reference because BF16 and FP16 differ throughout the entire forward pass, not just at the head. What it provides is an empirical guarantee: on the evaluation set, prompts with tight margins are the ones that diverge, and covering them with FP32 head recomputation recovers most of the agreement loss. The body error (accumulated over 22 layers) accounts for a small residual fraction of divergence that the head-only fix does not address.

---

## Experimental Findings

Across six models (1.1B–7B, four families) and three benchmarks:

- **Divergence rate:** 49–100% of prompts produce different BF16 vs. FP16 greedy outputs; the rate varies by model family and benchmark but is never negligible.
- **Cascade effect:** A single first-token flip cascades into full trajectory divergence in over 80% of cases.
- **Error-propagation analysis:** Accumulated body error over 22 transformer layers does not differentiate flipping from non-flipping steps; the top-two margin at the LM head is the dominant predictor.
- **Selective FP32 head recomputation:**
  - A10G GPU: +22–36 percentage-point exact agreement recovery.
  - L4 and A100 GPUs: +12–21 pp recovery (smaller because baseline divergence rates differ by architecture).
  - Latency overhead: ±4% in low-batch single-stream inference.

---

## Ablations and Interpretation

- **Threshold sensitivity:** Agreement recovery is robust to threshold choice across a 5× range of $\tau$; setting $\tau$ too low misses tight-margin steps, setting it too high incurs unnecessary FP32 recomputations.
- **Full FP32 inference:** Achieves near-zero divergence but costs 2× memory and ~1.8× latency — impractical for production.
- **Body recomputation:** Running the entire transformer in FP32 only at flip-predicted steps adds disproportionate overhead (the body is much larger than the head) for marginal additional gain.
- **Model family variation:** Models with larger vocabulary sizes have smaller average top-two margins (more competition per step) and higher divergence rates, suggesting that the phenomenon worsens as vocabulary size grows.
- **Benchmark dependence:** Open-ended generation (TriviaQA) shows higher divergence than constrained generation (MMLU multiple choice), as cascading divergence has more space to propagate.

---

## Reference

Gaoyuan Du, Anam Nawaz Khan, Rex Zhou, Xiaoyang Liu, Deepayan Chakrabarti, Fnu Suya, Xueping Li. **Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference**. arXiv:2609.26621, September 2026.  
https://arxiv.org/abs/2609.26621
