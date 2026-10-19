# OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding

**arXiv:** 2610.18829  
**Venue:** arXiv preprint, October 2026  
**Authors:** Xuyang Liu, Shihao Wang, Xinyu Wei, Minghan Li, Zeqing Wang, Zuhao Yang, Zichen Wen, Lei Zhang  
**Affiliations:** Hong Kong Polytechnic University, OPPO Research Institute  
**Date:** October 13, 2026

---

## One-Line Summary

OmniCompass introduces a paired-setting evaluation protocol for omni-modal video understanding that reveals deep inconsistencies in today's best models—testing whether they genuinely ground queries in visual evidence or merely match instruction patterns.

---

## Problem

Contemporary video benchmarks evaluate language comprehension or visual understanding in isolation. With the arrival of omni-modal models capable of processing video, audio, and text simultaneously, a gap has emerged: no benchmark systematically tests whether a model can ground a query in joint audio-visual evidence, or whether it simply pattern-matches the text instruction format to arrive at a correct answer by chance. This gap masks a fundamental limitation: models that pass text-only evaluation may fail when the same query is paired with contradictory visual evidence, and vice versa.

---

## Key Observation: Paired Settings Expose Pattern-Matching

OmniCompass evaluates each instance under two conditions:
1. **Text-only:** The query is posed without any visual or audio context.
2. **Text+image/text+audio:** The same query is posed with the relevant visual or audio evidence.

A model that truly understands the target should respond consistently (and correctly) under both conditions, or should improve under the richer condition. Inconsistent responses between the two conditions—answering correctly under text-only but incorrectly with visual evidence, or vice versa—indicate surface-level pattern matching rather than genuine grounding.

---

## Why Existing Approaches Fall Short

- **Standard VQA benchmarks:** Test single-image comprehension; do not cover video-level temporal or spatial grounding.
- **Text-only LLM benchmarks:** Ignore visual and audio modalities entirely; cannot measure multimodal grounding.
- **Existing video QA benchmarks (e.g., MVBench, VideoMME):** Focus on visual understanding; do not include audio-dependent tasks or the paired-setting probe.
- **Audio-visual benchmarks:** Limited to small datasets; do not cover temporal or spatial grounding with the structured annotation depth OmniCompass provides.

---

## Core Method: OmniCompass Design

**Two evaluation tracks:**

**OmniCompass-VL** (Vision-Language Track):
- 820 videos, 2,460 instances.
- Questions require visual evidence for correct answers.
- Eight task categories: five perception (object recognition, action recognition, attribute identification, counting, OCR) and three reasoning (causal, commonsense, temporal reasoning).
- Temporal grounding (TG), spatial grounding (SG), and question answering (QA) subtasks.

**OmniCompass-AV** (Audio-Visual Track):
- 240 videos, 720 instances drawn from narrative-rich, audio-dense content.
- Questions require joint audio-visual evidence.
- Five task categories: three perception (speech recognition, sound identification, speaker localization) and two reasoning (audio-visual causal reasoning, event coordination).
- Same TG/SG/QA structure as the VL track.

**Paired evaluation protocol:**
For every instance, each model is evaluated twice: once with text alone and once with text plus the target image frame or audio clip. The gap between the two scores measures reliance on visual/audio evidence versus instruction-format matching.

---

## Technical Formulation

Let $q_i$ be the $i$-th query, $v_i$ be the corresponding visual/audio evidence, and $a_i^*$ be the ground-truth answer. For each model $\mathcal{M}$:

$$s_i^{\text{text}} = \mathbf{1}[\mathcal{M}(q_i) = a_i^*]$$
$$s_i^{\text{multi}} = \mathbf{1}[\mathcal{M}(q_i, v_i) = a_i^*]$$

**Consistency score:**
$$C(\mathcal{M}) = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[s_i^{\text{text}} = s_i^{\text{multi}}]$$

A high $C(\mathcal{M})$ with low $\sum_i s_i^{\text{multi}}$ indicates consistent failure; a high $C(\mathcal{M})$ with high $\sum_i s_i^{\text{multi}}$ indicates reliable visual grounding. A low $C(\mathcal{M})$ regardless of accuracy indicates format sensitivity.

**Grounding precision** is measured separately for temporal (overlap with ground-truth time segment) and spatial (IoU with ground-truth bounding box) tasks.

---

## Learning or Inference Procedure

OmniCompass is a benchmark, not a trained model. The protocol:

1. Collect videos with expert-annotated temporal segments, spatial bounding boxes, and question-answer pairs.
2. Construct paired instances: text-only and text+evidence versions of each question.
3. Run each evaluated model in zero-shot mode on all instances.
4. Compute per-track accuracy, grounding precision, and consistency scores.
5. Report results stratified by task category and model family.

---

## What the Paper Claims

- Open-source models still trail proprietary counterparts on both tracks, particularly on audio-visual understanding.
- Precise temporal and spatial grounding remains difficult even for leading models.
- Paired-setting inconsistencies are widespread, indicating that many models arrive at correct text-only answers by exploiting question format rather than visual content.
- The audio-visual track reveals a larger performance gap between open-source and proprietary models than the vision-language track.

---

## Experimental Findings

- **Models evaluated:** 50 MLLMs, spanning open-source (Qwen-VL, LLaVA variants, InternVL, etc.) and proprietary (GPT-4o, Claude models, Gemini models).
- **OmniCompass-VL accuracy:** Proprietary models average ~15–20 points higher than open-source on QA subtasks.
- **OmniCompass-AV accuracy:** The gap between proprietary and open-source models widens; several open-source models perform near chance on the audio-visual reasoning subtask.
- **Consistency scores:** Majority of models show consistency scores below 70%, meaning over 30% of instances produce different correctness outcomes under the two evaluation conditions.
- **Grounding precision:** Temporal IoU and spatial IoU scores are low across all model families, confirming that grounding—not just QA—is the limiting capability.

---

## Ablations and Interpretation

- **Paired-gap analysis:** Models with the smallest text-vs-multimodal accuracy gap tend to be the most accurate overall, confirming that genuine grounding correlates with performance.
- **Category breakdown:** Temporal reasoning and causal reasoning show the lowest consistency, suggesting models rely heavily on surface-level cues for these categories.
- **Audio-visual subtask:** Models that excel at VL tasks do not uniformly transfer to AV tasks; audio-visual reasoning appears to require explicitly joint multimodal processing.
- **Benchmark calibration:** The OmniCompass team verified inter-annotator agreement (Cohen's κ > 0.8) for all task categories, and adversarially filtered questions solvable from text alone.

---

## Reference

Xuyang Liu, Shihao Wang, Xinyu Wei, Minghan Li, Zeqing Wang, Zuhao Yang, Zichen Wen, Lei Zhang. **OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding**. arXiv:2610.18829. October 2026.  
https://arxiv.org/abs/2610.18829
