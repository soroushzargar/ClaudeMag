# Mask-Aware Policy Gradients for Diffusion Language Models

**arXiv:** 2607.15200  
**Date:** July 17, 2026  
**Authors:** Haran Raajesh, Kulin Shah, Adam Klivans, Philipp Krähenbühl  
**Affiliation:** University of Texas at Austin  
**URL:** https://arxiv.org/abs/2607.15200  
**Venue:** COLM 2026

---

## Headline

Mask-Aware Policy Gradients resolve a fundamental mismatch in RL training for masked diffusion language models: existing methods ignore the masking decisions in the policy gradient, producing biased updates; the correct gradient decomposes into a token term and a separate masking term, improving GSM8K from 79.3% to 87.1% and MBPP from 41.2% to 53.4%.

---

## Key Findings

- MDLM generation involves two interleaved decisions at each step: what token to place at each unmasked position and which positions to remask for the next step; ignoring the masking decision produces a biased policy gradient.
- The correct policy gradient for MDLMs decomposes into a token prediction term and a masking schedule term; jointly optimizing both significantly outperforms approaches that optimize token prediction alone.
- The method achieves 87.1% on GSM8K and 53.4% on MBPP, state-of-the-art results for diffusion language models on these benchmarks.
- The masking term is a previously overlooked source of variance in MDLM policy gradients; controlling it improves training stability and final performance.
- The two-stage action MDP formulation is theoretically grounded: the decomposition is exact under the MDLM generative process, not an approximation.

---

## Methodology

Masked Diffusion Language Models generate text by iteratively unmasking tokens over $T$ steps. At step $t$, the model observes a partially masked sequence $x_t$ and:
1. Predicts the unmasked tokens at each masked position (token action $a^{\text{tok}}_t$).
2. Decides which of the currently unmasked positions to re-mask for the next step (masking action $a^{\text{mask}}_t$).

Standard RL for MDLMs treats each step as a one-stage action (only $a^{\text{tok}}_t$), dropping $a^{\text{mask}}_t$ from the policy gradient. TRACE-MDLM introduces a two-stage MDP where both actions are part of the policy, and the policy gradient is:

$$\nabla_\theta J = \mathbb{E}_\pi \left[ \sum_t \hat{A}_t \left( \nabla_\theta \log \pi_\theta(a^{\text{tok}}_t \mid x_t) + \nabla_\theta \log \pi_\theta(a^{\text{mask}}_t \mid x_t, a^{\text{tok}}_t) \right) \right]$$

where $\hat{A}_t$ is the advantage estimate at step $t$.

---

## Technical Details

- **MDLM base model:** The authors build on MDLM-3B, a masked diffusion model trained on The Pile.
- **Token action policy:** The standard forward pass predicts the denoised token at each masked position via the MDLM network.
- **Masking action policy:** A lightweight two-layer MLP head takes the token-step hidden states and outputs a masking probability per position; trained jointly with the denoising network.
- **Advantage estimation:** Generalized Advantage Estimation (GAE) with $\lambda = 0.95$; the value function is a separate head on the MDLM encoder.
- **Reward:** Exact-match rewards for GSM8K (math) and MBPP (code); no intermediate process rewards are used.
- **Baseline comparison:** Compared against REINFORCE with token-only gradient, SPG (Sandwiched Policy Gradient), and GRPO applied to MDLMs.

---

## Experimental Findings

| Method | GSM8K | MBPP |
|---|---|---|
| MDLM-3B (no RL) | 52.1% | 28.7% |
| Token-only REINFORCE | 79.3% | 41.2% |
| SPG (2510.09541) | 81.4% | 44.7% |
| **Mask-Aware PG (ours)** | **87.1%** | **53.4%** |

The masking term alone (without changing token gradient) contributes approximately 3–4 percentage points of improvement; the joint optimization contributes the remaining 4–5 points.

Training stability is significantly improved: token-only methods exhibit gradient variance spikes every 50–100 steps; Mask-Aware PG has 40% lower gradient norm variance across training.

---

## Ablations and Interpretation

- **Token gradient only (ablation):** Removing the masking term reproduces the baseline; confirms the masking term is the source of improvement.
- **Masking term only (ablation):** Training with only the masking gradient (no token gradient) fails to converge, confirming both terms are necessary.
- **Masking schedule analysis:** Models trained with Mask-Aware PG learn to unmask tokens in a more semantically coherent order (nouns and verbs early, modifiers late in math problems), suggesting the masking term teaches a meaningful unmasking curriculum.
- **Scaling:** The advantage of Mask-Aware PG grows with problem difficulty; for easy GSM8K problems, the gap is small, but for hard multi-step reasoning problems it is 8+ percentage points.

---

## Reference

Haran Raajesh, Kulin Shah, Adam Klivans, Philipp Krähenbühl. **Mask-Aware Policy Gradients for Diffusion Language Models.** COLM 2026 / arXiv:2607.15200, July 2026. https://arxiv.org/abs/2607.15200
