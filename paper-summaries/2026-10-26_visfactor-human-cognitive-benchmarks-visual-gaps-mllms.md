# Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs

**arXiv:** 2502.16435  
**Venue:** NeurIPS 2026  
**Authors:** Jen-Tse Huang, Dasen Dai, Jen-Yuan Huang, Youliang Yuan, Xiaoyuan Liu, Wenxuan Wang, Wenxiang Jiao, Pinjia He, Zhaopeng Tu, Haodong Duan  
**Affiliation:** CUHK-ARISE  
**Date:** February 2025 (NeurIPS 2026 version revised October 2026)

---

## One-Line Summary

VisFactor—a benchmark digitized from 20 human visual-cognition subtests across four domains—reveals that 23 frontier MLLMs achieve a best score of only 30.17 out of 100, exposing foundational visual gaps that conventional academic benchmarks fail to surface.

---

## Problem

Multimodal large language models (MLLMs) are evaluated on a growing battery of academic benchmarks—VQA, MMBench, MMMU, SeedBench—that test combinations of vision and language understanding. Yet these benchmarks are constructed by ML researchers and may implicitly favour models that can leverage language priors, common-sense reasoning, and visual-language alignment to compensate for weak raw visual perception. The cognitive science literature, by contrast, has decades of psychophysical assessment tools designed specifically to measure the low-level and mid-level visual processing capabilities of the human visual system, independent of language. Applying these tools to MLLMs may reveal visual deficiencies that aggregate accuracy on academic tasks conceals.

---

## Key Observation: Cognitive Psychology Batteries Are Human-Validated, Granular, and Transfer-Proof

The Factor-Referenced Cognitive Tests (FRCT) are standardized psychometric instruments designed to assess specific factors of human cognitive ability. The visual subset—FRCT-V—isolates four orthogonal dimensions of visual cognition: spatial reasoning, perceptual speed, visualization, and visual memory. These subtests were designed to be relatively language-free and to resist strategy-based shortcuts, making them ideal probes of raw visual computation.

---

## Why Existing Benchmarks Fall Short

- **Language-shortcut susceptibility:** Many VQA benchmarks can be partially answered from question text alone; models with strong language priors can score well without parsing the image.
- **Coarse granularity:** Aggregate scores mask which specific visual processes are failing; a model may excel at object recognition while being unable to mentally rotate a shape.
- **Artificial ceiling effects:** Frontier models saturate many existing benchmarks, leaving no room to distinguish capability gaps among top-tier systems.
- **Absence of psychometric grounding:** Existing benchmarks lack the normative calibration against human populations that enables meaningful comparison of model performance to human visual cognition.

---

## Core Method: VisFactor Benchmark

VisFactor digitizes 20 vision-centric subtests from the FRCT battery, covering four domains:

| Domain | Example Subtests | Visual Faculty Probed |
|--------|-----------------|----------------------|
| Spatial Reasoning | Card Rotations, Paper Folding | Mental rotation, spatial visualization |
| Perceptual Speed | Finding A's, Number Comparison | Rapid comparison, figure-ground segregation |
| Visualization | Surface Development, Form Board | 3D → 2D reasoning, shape completion |
| Visual Memory | Shape Memory, Position Memory | Short-term retention of visual configurations |

Each subtest is digitized as a multiple-choice or mark-verification visual question, preserving the psychometric structure of the original test (timed conditions are converted to accuracy-only for fair LLM evaluation). Each question is paired with a ground-truth answer verified against the original FRCT scoring key.

The final benchmark comprises items from all 20 subtests, balanced across the four domains, with a total score normalized to 100.

---

## Technical Formulation

Let $M$ be an MLLM and $\mathcal{Q} = \{(q_i, A_i, a_i^*)\}_{i=1}^N$ be the VisFactor question set (question image $q_i$, answer options $A_i$, correct answer $a_i^*$). The VisFactor score is:

$$\text{Score}(M) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[M(q_i, A_i) = a_i^*] \times 100$$

Domain subscores partition the sum over the four cognitive domains, enabling per-domain analysis. A random baseline scores approximately 25 (four-option multiple choice); the human norm from the original FRCT study is calibrated to approximately 80.

---

## Learning or Inference Procedure

VisFactor is a zero-shot evaluation benchmark. Models are queried with:
- The image depicting the visual reasoning task
- The multiple-choice options
- A minimal instruction string (e.g., "Which option correctly completes the pattern? Answer with a single letter.")

No chain-of-thought prompting is used by default; the authors also report a chain-of-thought variant where models are asked to explain their reasoning before answering.

---

## What the Paper Claims

- **Severe shortfall:** The best-performing model (frontier proprietary) scores 30.17/100; the average across 23 models is below 25.
- **Domain asymmetry:** Models perform best on perceptual speed tasks (simple binary comparisons) and worst on visualization tasks (3D-to-2D mental transformations), suggesting a specific deficit in spatial reasoning.
- **No clear advantage for larger models:** Scaling model size does not reliably improve VisFactor scores, unlike most academic visual benchmarks where scale helps.
- **Language shortcuts do not help:** Unlike VQA tasks, VisFactor items cannot be answered by reading the question text; performance reflects genuine visual computation.
- **Chain-of-thought does not close the gap:** Prompting models to reason step-by-step before answering provides at most a marginal improvement, and sometimes degrades performance on perceptual speed tasks.

---

## Experimental Findings

- **Models evaluated (v3):** 23 frontier MLLMs, including GPT-4V/4o, Gemini-1.5/2.0, Claude-3.5, LLaMA-3-Vision, Qwen-VL-Max, InternVL-2.5, and others.
- **Best model:** Achieves 30.17/100; human norm is ~80.
- **Domain breakdown:** Perceptual speed ≈ 45%; spatial reasoning ≈ 28%; visualization ≈ 20%; visual memory ≈ 22% (approximate from v3).
- **Saturation not reached:** No model approaches the human norm on any domain, confirming persistent capability gaps.

---

## Ablations and Interpretation

- **Image-only vs. text+image:** Removing the image reduces performance to chance (25%), confirming models do engage with visual input.
- **Answer option order:** Shuffling options does not significantly change results, ruling out position bias as a main source of error.
- **Subtest difficulty analysis:** Items requiring mental rotation of complex 3D figures are among the hardest; items requiring simple symbol-matching are comparatively easier, matching the psychometric structure.

---

## Reference

Jen-Tse Huang, Dasen Dai, Jen-Yuan Huang, Youliang Yuan, Xiaoyuan Liu, Wenxuan Wang, Wenxiang Jiao, Pinjia He, Zhaopeng Tu, Haodong Duan. **Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs**. arXiv:2502.16435. NeurIPS 2026.  
https://arxiv.org/abs/2502.16435
