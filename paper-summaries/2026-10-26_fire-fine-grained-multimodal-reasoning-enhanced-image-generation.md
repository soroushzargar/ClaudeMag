# FiRe: Fine-grained Multimodal Reasoning for Enhanced Image Generation

**arXiv:** 2604.13491  
**Venue:** NeurIPS 2026  
**Authors:** Yongjin Kim, Yoonjin Oh, Yerin Kim, Hyomin Kim, Jeeyoung Yun, Yujung Heo, Minjun Kim, Sungwoong Kim  
**Affiliation:** KU-AGI (Korea University)  
**Date:** April 2026 (NeurIPS 2026 camera-ready October 2026)

---

## One-Line Summary

FiRe decomposes a text prompt into visual requirements, checks each requirement against the generated image with an MLLM, and refines the image locally—paired with FiRe-GRPO, a step-level RL objective—to achieve consistent compositional image generation gains at NeurIPS 2026.

---

## Problem

Text-to-image generative models frequently fail on compositional prompts that specify multiple simultaneous visual constraints—color, count, spatial arrangement, and attribute binding. Current reasoning-augmented approaches address this through whole-prompt elaboration or single-image feedback, which cannot pinpoint which individual requirement is violated or guide a targeted local correction. As a result, iterative refinement loops tend to fix one visual element while inadvertently degrading another.

---

## Key Observation: Decomposing Prompts into Checkable Requirements Enables Targeted Refinement

A compositional prompt can be parsed into a list of atomic visual requirements, each testable independently. A multimodal LLM can answer "Is requirement R satisfied in this image region?" with high reliability even when holistic prompt-satisfaction is hard to assess. Grounding corrections at the requirement level rather than the image level prevents the fix-one-break-another failure mode.

---

## Why Existing Approaches Fall Short

- **Prompt augmentation (chain-of-thought elaboration before generation):** Improves global coherence but does not adapt to failures in a specific generated image; the same elaboration may work for one seed but not another.
- **Whole-image feedback (DALL-E Reflect, ImageReward):** Produces a scalar or sentence-level critique that is too coarse to guide local pixel-level corrections.
- **General RLHF for image generation:** Optimizes for a reward that aggregates all visual properties, making it hard to distinguish which property drives the gradient signal or to balance multiple equally important constraints.

---

## Core Method

**FiRe pipeline:**

1. **Requirement decomposition:** An MLLM parses the user prompt into a structured list of visual requirements $\{r_1, \ldots, r_n\}$, each describing a testable visual property (e.g., "a red cup", "two dogs", "cup to the left of the dog").
2. **Generation:** A base text-to-image model generates an initial image $I_0$.
3. **Fine-grained check:** For each requirement $r_i$, the MLLM checks whether $r_i$ is satisfied in the relevant region of $I_0$, producing a binary verdict and a natural-language diagnosis.
4. **Local refinement:** Unsatisfied requirements drive targeted edits to the corresponding image region via an inpainting or latent-space edit step, leaving the rest of the image unchanged.
5. **Iteration:** Steps 3–4 repeat until all requirements are satisfied or a budget is exhausted.

**FiRe-GRPO:**

The iterative check-refine cycle is framed as a multi-step reasoning trajectory. FiRe-GRPO assigns a dense reward to each reasoning step (check verdict + refinement action) rather than only to the final image. The reward for a step involving requirement $r_i$ is based on the improvement in $r_i$'s satisfaction probability between $I_{t-1}$ and $I_t$, measured by the MLLM. Group Relative Policy Optimization (GRPO) is then used to fine-tune the unified MLLM backbone that both checks and guides refinement.

---

## Technical Formulation

Let $\mathcal{V}_i(I) \in [0,1]$ denote the MLLM's estimate of the probability that requirement $r_i$ is satisfied in image $I$. At step $t$, the refinement selects the least-satisfied requirement $i^* = \arg\min_i \mathcal{V}_i(I_{t-1})$ and applies a latent edit $\delta_t$ guided by the MLLM's diagnosis, yielding $I_t$.

The step-level GRPO reward for the $t$-th refinement step is:

$$\rho_t = \mathcal{V}_{i^*}(I_t) - \mathcal{V}_{i^*}(I_{t-1})$$

The policy gradient update maximizes:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_t \rho_t \cdot \log \pi_\theta(\delta_t \mid I_{t-1}, r_{i^*})\right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $\pi_{\text{ref}}$ is a frozen reference policy and $\beta$ controls the KL regularization.

---

## Learning or Inference Procedure

**Training (FiRe-GRPO):**
1. Sample a prompt, decompose into requirements, generate an initial image.
2. Run the check-refine cycle for $T$ steps; collect step rewards $\rho_t$.
3. Compute policy gradient with the GRPO objective and update the MLLM backbone.
4. Use the 300-step checkpoint (FiRe-300Step) for evaluation, balancing refinement quality with inference cost.

**Inference (FiRe):**
1. Parse the prompt into requirements.
2. Generate with the base T2I model.
3. Iteratively check and locally refine until the budget is reached or all requirements are satisfied.
4. Return the final image.

---

## What the Paper Claims

- **Compositional gains:** Consistent improvement over text-to-image baselines on compositional benchmarks (T2I-CompBench, GenAI-Bench), with the largest gains on multi-constraint prompts.
- **Local refinement advantage:** Targeted per-requirement edits avoid degrading already-satisfied visual elements, unlike whole-image refinement.
- **FiRe-GRPO benefit:** Step-level rewards lead to faster convergence during training and better refinement policies at inference compared to terminal-only reward formulations.

---

## Experimental Findings

- **Benchmarks:** T2I-CompBench (attribute binding, spatial relationships, non-spatial relationships), GenAI-Bench (compositional), and qualitative evaluations.
- **Baselines:** SD-XL, DALL-E 3, prompt-augmented variants, and whole-image feedback methods.
- **Key results:** FiRe consistently improves attribute binding and spatial relationship accuracy; gains are largest when the prompt contains 3+ simultaneous constraints.
- **Ablations:** Removing the local refinement step and using global image edits recovers the fix-one-break-another failure mode; removing FiRe-GRPO and using supervised fine-tuning degrades refinement policy quality.

---

## Reference

Yongjin Kim, Yoonjin Oh, Yerin Kim, Hyomin Kim, Jeeyoung Yun, Yujung Heo, Minjun Kim, Sungwoong Kim. **FiRe: Fine-grained Multimodal Reasoning for Enhanced Image Generation**. arXiv:2604.13491. NeurIPS 2026.  
https://arxiv.org/abs/2604.13491
