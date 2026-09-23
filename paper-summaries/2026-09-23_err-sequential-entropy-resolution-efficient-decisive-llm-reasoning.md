# ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning

**arXiv:** 2608.28771  
**Venue:** Findings of EMNLP 2026  
**Authors:** Juxin Chen et al.  
**Date:** August–September 2026

---

## One-Line Summary

Correct CoT traces have more frequent, larger token-level entropy drops than incorrect ones; ERR+ rewards these drops in Phase 1 and compresses response length in Phase 2, improving both reasoning accuracy and efficiency.

---

## Problem

Reinforcement learning with verifiable rewards (RLVR) has become the dominant paradigm for training large reasoning models (LRMs). These models produce extended chain-of-thought (CoT) traces during a "thinking" phase before emitting a final answer. While RLVR reliably improves accuracy, it often produces verbose, repetitive, or exploratory traces that waste inference compute.

Crucially, existing RLVR objectives reward correct final answers but say nothing about the internal quality of the trace—whether the model is making decisive, well-directed inferences or wandering through redundant alternatives.

---

## Key Observation: Entropy Drops as a Signal of Decisive Reasoning

Through systematic empirical analysis across multiple model families and task types, the authors identify a consistent, previously-unnoticed pattern:

**Correct reasoning traces exhibit more frequent and larger token-level entropy drops within the thinking phase than incorrect traces.**

Token-level entropy is defined as $H_t = -\sum_v p_\theta(v | x_{<t}) \log p_\theta(v | x_{<t})$, the Shannon entropy of the next-token distribution at position $t$. An entropy *drop* at position $t$ is $\Delta H_t = H_{t-1} - H_t > 0$—a moment when the model becomes more decisive.

The observation suggests that decisive, targeted reasoning (reflected by frequent entropy drops) correlates with correctness, while verbose or uncertain traces (fewer, smaller drops) correlate with errors.

---

## Why Existing Approaches Fall Short

- **RLVR with binary outcome reward** does not distinguish decisive from verbose correct traces; it rewards both equally
- **Length penalty alone** discourages long traces indiscriminately, potentially cutting useful reasoning
- **Token-level KL regularization** controls entropy globally but does not specifically reward decisive resolution *during* the thinking phase

ERR+ specifically rewards the *pattern* of entropy change across the trace, not its aggregate length or its final-state entropy.

---

## Core Method: Two-Phase RLVR

**Phase 1 — Entropy Relief Reward (ERR).** Train with RLVR where the reward includes an ERR bonus:

$$r_{\text{ERR}} = r_{\text{base}} + \beta \cdot \text{ERR}(\tau)$$

where $r_{\text{base}}$ is the standard correctness reward and:

$$\text{ERR}(\tau) = \frac{\sum_{t \in \mathcal{T}_{\text{think}}} \max(0, \Delta H_t)}{\log |\tau|} \cdot \mathbf{1}[a = a^*]$$

The gating by $\mathbf{1}[a = a^*]$ ensures that entropy drops are only rewarded in *correct* traces. The log-length normalization prevents the model from earning ERR rewards simply by generating longer traces with more drops in absolute terms.

**Phase 2 — Length Compression.** The Phase-1 model is further trained with a group relative policy optimization (GRPO) objective that adds a length-relative penalty:

$$r_{\text{Phase2}} = r_{\text{base}} - \gamma \cdot \frac{|\tau| - \bar{|\tau|}_{\text{group}}}{\bar{|\tau|}_{\text{group}}}$$

where $\bar{|\tau|}_{\text{group}}$ is the mean length of traces in the current rollout group. This rewards brevity *relative to peers*, not absolutely, avoiding reward collapse to degenerate short outputs.

---

## Technical Formulation

Let $x_{<t}$ be the token sequence up to position $t$ during the thinking phase $\mathcal{T}_{\text{think}}$. Token entropy:

$$H_t = -\sum_{v \in \mathcal{V}} p_\theta(v \mid x_{<t}) \log p_\theta(v \mid x_{<t})$$

Cumulative entropy drop:

$$\text{CED}(\tau) = \sum_{t \in \mathcal{T}_{\text{think}}} \max(0, H_{t-1} - H_t)$$

Log-normalized entropy relief:

$$\text{ERR}(\tau) = \frac{\text{CED}(\tau)}{\log(|\tau| + 1)} \cdot \mathbf{1}[a(\tau) = a^*]$$

Phase 1 policy gradient:

$$\nabla_\theta \mathcal{L}_1 = -\mathbb{E}_\tau\!\left[(r_{\text{base}} + \beta \cdot \text{ERR}(\tau)) \cdot \nabla_\theta \log p_\theta(\tau)\right]$$

---

## Learning Procedure

1. Start from a base reasoning model fine-tuned on CoT data.
2. **Phase 1:** Run RLVR with $r_{\text{ERR}}$ for $N_1$ steps; monitor accuracy and mean CED.
3. **Phase 2:** Continue from Phase-1 checkpoint with $r_{\text{Phase2}}$ for $N_2$ steps; monitor accuracy and trace length.
4. Evaluate on held-out benchmarks with a fixed inference budget (no search).

---

## What the Paper Claims

ERR+ improves final-answer accuracy and reduces trace length simultaneously on five math and STEM reasoning benchmarks. The sequential (two-phase) structure is essential: applying length compression in Phase 1 degrades accuracy because it prevents the model from developing decisive reasoning patterns first.

---

## Experimental Findings

- Evaluated on AIME 2024, AMC 2023, MATH-500, GPQA-Diamond, and a STEM multi-step dataset
- Accuracy improves by 2–5 percentage points over RLVR-only baselines
- Mean trace length decreases by 15–25% relative to Phase-1 baseline
- ERR+ outperforms length-penalty-only methods at matched trace length

---

## Ablations and Interpretation

- **Phase 1 only:** Accuracy improvement without length reduction; confirms ERR bonus drives decisive reasoning
- **Phase 2 only (no Phase 1):** Length reduction with accuracy degradation; confirms Phase 1 is foundational
- **No log-length normalization:** Accuracy improvement but trace length *increases* (model learns to earn ERR by generating more drops absolutely)
- **No correctness gating:** Accuracy drops; entropy drops in incorrect traces are encouraged, destabilizing reasoning
- **$\beta$ sweep:** Optimal $\beta \in [0.2, 0.5]$; too large collapses entropy and hurts exploration

---

## Reference

Juxin Chen et al. **ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning**. arXiv:2608.28771. Findings of EMNLP 2026.  
https://arxiv.org/abs/2608.28771
