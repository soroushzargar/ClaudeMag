# Test-Time Speculation

**arXiv:** 2605.09329  
**Venue:** NeurIPS 2026  
**Authors:** Avinash Kumar, Sujay Sanghavi, Poulami Das  
**Affiliation:** The University of Texas at Austin  
**Date:** May 2026 (NeurIPS 2026 camera-ready October 2026)

---

## One-Line Summary

Test-Time Speculation (TTS) adapts the draft model continuously during inference using target-model evaluations already computed by the verification step, at zero added cost, and sustains high acceptance lengths throughout long chain-of-thought generation—improving acceptance by up to 72% over static state-of-the-art speculators.

---

## Problem

Speculative decoding accelerates LLM inference by having a small, fast draft model generate candidate tokens that a large target model verifies in parallel. The speedup depends critically on the acceptance length—how many draft tokens the target accepts before rejection. In practice, the acceptance length of trained speculators such as EAGLE-3, DFlash, and PARD decays as generation continues: it starts high but falls toward 1 after a few thousand output tokens, eliminating the speedup for long chain-of-thought (CoT) and reasoning outputs.

The root cause is a training distribution mismatch: all existing speculators are trained offline on short sequences, but at inference they must predict continuations of sequences much longer than any training example. The draft model's internal distribution drifts from the target's as the context grows.

---

## Key Observation: The Verification Step Already Computes Exactly What Is Needed for Online Training

In every speculative decoding step, the target model evaluates the full draft sequence to produce acceptance/rejection decisions. These evaluations provide token-level probability labels from the target model at the current context length—precisely the student-teacher signal needed to adapt the draft. Conventional speculative decoding discards this information after verification; TTS re-uses it to update the draft model at no additional forward pass cost.

---

## Why Existing Approaches Fall Short

- **Offline training on short sequences:** EAGLE-3, DFlash, PARD, and similar systems train once on datasets of typical generation lengths. Long inference-time sequences fall outside the training distribution, and the draft model has no mechanism to adjust.
- **Static draft at inference:** Once trained, the draft model is frozen. It cannot recover from distributional drift as the context grows.
- **Re-training or fine-tuning approaches:** Online adaptation methods for NLP typically require periodic re-training runs, which involve additional inference calls and interrupt the generation pipeline.
- **Longer training data:** Simply including longer sequences in offline training is expensive and does not capture the inference-time distribution, which depends on the specific prompt and intermediate reasoning steps.

---

## Core Method: Online Draft Distillation During Generation

**TTS algorithm:**

At each speculative decoding step $t$:

1. **Draft:** The draft model $\pi_\theta$ generates $k$ candidate tokens from context $x_{<t}$.
2. **Verify:** The target model $\pi^*$ evaluates the candidates and accepts/rejects each token, as in standard speculative decoding.
3. **Distill:** Using the token-level target probabilities $\pi^*(x_j \mid x_{<j})$ for $j \in \{t, \ldots, t+k\}$ obtained in step 2, compute a KL distillation loss $\mathcal{L}_t$ between $\pi_\theta$ and $\pi^*$ at those positions.
4. **Update:** Take one gradient step on $\pi_\theta$ to minimize $\mathcal{L}_t + \beta \cdot \Omega(\pi_\theta, \pi_{\theta_{t-1}})$, where $\Omega$ is a proximity regularizer (e.g., squared parameter distance) that prevents the draft model from diverging too quickly.
5. **Generate:** Continue generation from the accepted token position.

The draft model is typically a small, distilled version of the target (e.g., 1–3B parameters), so the gradient step is cheap. The total added compute per step is one optimizer update on a small model, dominated by the target model's existing verification pass.

---

## Technical Formulation

Let $\pi^*$ denote the target model and $\pi_\theta$ the draft model with parameters $\theta$. At step $t$, the distillation loss over the $k$ draft positions is:

$$\mathcal{L}_t(\theta) = \frac{1}{k} \sum_{j=t}^{t+k} D_{\text{KL}}\!\left(\pi^*(\cdot \mid x_{<j}) \;\|\; \pi_\theta(\cdot \mid x_{<j})\right)$$

The proximity regularizer prevents catastrophic forgetting of the base draft distribution:

$$\Omega(\pi_\theta, \pi_{\theta_{t-1}}) = \|\theta - \theta_{t-1}\|_2^2$$

The combined update minimizes:

$$\theta_t = \theta_{t-1} - \eta \nabla_\theta\!\left[\mathcal{L}_t(\theta_{t-1}) + \beta \cdot \Omega(\theta_{t-1}, \theta_{t-2})\right]$$

with learning rate $\eta$ and regularization weight $\beta$. A cosine learning rate schedule is used over the generation length; $\beta$ is set so that the total parameter shift over a full generation does not exceed a small threshold.

---

## Learning or Inference Procedure

**Standard speculative decoding (baseline):**
- Draft model is frozen at inference; acceptance length degrades with context length.

**TTS (proposed):**
1. Initialize draft model from any pre-trained speculator (EAGLE-3, DFlash, PARD, etc.).
2. During generation, after each verification step, compute $\mathcal{L}_t$ and take one gradient step on the draft model.
3. No external data or target-model forward passes beyond those already required by verification.
4. The updated draft model produces better candidates at longer contexts, sustaining acceptance length throughout generation.

---

## What the Paper Claims

- **Acceptance length improvement:** TTS improves acceptance length by up to 72% over the best static speculator, with average improvement of 41% across Qwen-3, Qwen-3.5, and Llama 3.1 model families.
- **Gains grow with length:** The improvement is small for short outputs (<1K tokens) and large for long CoT outputs (>5K tokens), exactly the regime where static speculators fail.
- **Zero added inference cost:** The single gradient step on the small draft model is negligible compared to the target model's verification pass.
- **Compatible with existing speculators:** TTS is a drop-in wrapper; it does not require retraining the base speculator or the target model.

---

## Experimental Findings

- **Models tested:** Qwen-3-8B with Qwen-3-1.5B draft, Qwen-3.5-32B with Qwen-3.5-7B draft, Llama-3.1-70B with Llama-3.1-8B draft.
- **Base speculators:** EAGLE-3, DFlash, PARD, and vanilla token-level distillation.
- **Benchmarks:** MATH-500 (long mathematical reasoning), HumanEval+ (long code generation), and a custom long-document summarization task.
- **Key result:** All static speculators show acceptance length decay; TTS applied to each of them recovers and maintains near-peak acceptance length throughout generation.
- **Wall-clock speedup:** TTS translates acceptance length gains to real speedup on A100 hardware; at 5K+ token generations, TTS achieves 2.1–3.4× over the target model, versus 1.2–1.8× for static speculators at the same length.

---

## Ablations and Interpretation

- **No proximity regularizer:** Without $\Omega$, the draft model overfits to the current context and loses the ability to generalize to the next step, reducing acceptance length by 18%.
- **Multiple gradient steps per verification:** More than one step per verification round does not help and slightly degrades speed by increasing the per-step overhead.
- **Fixed learning rate vs. cosine schedule:** A cosine decay schedule improves stability and final performance by ~5% average acceptance length.
- **Initialization from different base speculators:** TTS consistently improves over the base speculator it initializes from; EAGLE-3 + TTS achieves the best absolute acceptance length.

---

## Reference

Avinash Kumar, Sujay Sanghavi, Poulami Das. **Test-Time Speculation**. arXiv:2605.09329. NeurIPS 2026.  
https://arxiv.org/abs/2605.09329
