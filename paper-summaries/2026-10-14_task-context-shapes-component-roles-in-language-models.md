# Task Context Shapes Component Roles in Language Models

**arXiv:** 2610.13895  
**Venue:** arXiv preprint (October 2026)  
**Authors:** Sovesh Mohapatra, Dani S Bassett  
**Date:** October 9, 2026

---

## One-Line Summary

The same attention head serves different functional roles depending on task context; across five model families, task switches redistribute influence among components even when the prompt structure is identical, and early activation edits can redirect which fact a model reads without altering the instruction.

---

## Problem

Mechanistic interpretability research has identified individual attention heads and MLP neurons that appear to be "responsible" for specific facts or behaviors—induction heads for copying, name-mover heads for indirect object identification, etc. But almost all such work studies heads in isolation under a fixed task.

The question this paper poses is: **does an attention head that is responsible for a behavior under one task context remain responsible for the same behavior under a different task context, when the structural form of the prompt is held constant?**

If component roles are context-dependent, then explanations derived from single-task probing may not transfer and circuit-level interventions may have unintended cross-task consequences.

---

## Key Observation: Task-Dependent Influence Redistribution

The authors construct minimal-pair prompts called **graph facts**: pairs of prompts that differ only in the task instruction (e.g., "Complete: Paris is the capital of ___" vs. "Translate to French: Paris is the capital of France") while the factual content remains identical.

Measuring influence via activation patching (causal tracing), they find:

- **Same heads, different influence magnitudes.** A head that contributes the highest influence in Task A contributes substantially less in Task B, even though the factual content is unchanged.
- **Monotone redistribution.** The influence profile across layers shifts monotonically with task type; retrieval-heavy tasks concentrate influence in early-to-middle layers, while generation-heavy tasks shift it to later layers.
- **Predictable from response maps.** A linear response map $R_{ij}$ predicts the change in head $j$'s output at layer $\ell$ from an activation edit at head $i$ in layer $\ell' < \ell$. These maps are task-specific.

---

## Why Existing Approaches Fall Short

- **Single-task circuit tracing:** Identifies components responsible for a behavior in one setting; does not predict how those components behave under a different task with the same factual content.
- **Universal probes:** Train linear probes on a fixed dataset to locate where facts are stored; miss the dynamic routing that occurs when the task changes.
- **Attention pattern visualization:** Shows where a head attends but not how its output influences downstream layers; influence requires causal intervention, not just observation.

---

## Core Mechanism

The paper's methodology builds on activation patching (Meng et al. 2022) with three extensions:

1. **Graph-fact pairs:** Prompts are controlled so that the factual entity (e.g., a country capital) appears in the same token position across tasks, isolating task context from syntactic structure.

2. **Cross-task influence comparison:** For each head $(h, \ell)$, the indirect effect (IE) is measured as the change in the probability of the correct answer token when activations are patched from a clean run to a corrupted run:
   $$\text{IE}(h, \ell) = p_{\text{patched}}(a^* | x) - p_{\text{corrupted}}(a^* | x)$$

3. **Response maps:** A response map $R^{(t)}_{(h,\ell) \to (h',\ell')}$ is a $d \times d$ matrix that predicts $\Delta a_{h',\ell'}$ from $\Delta a_{h,\ell}$ under task $t$. It is estimated by linear regression over a held-out set of activation pairs.

---

## Technical Formulation

For task $t$, let $\{x^{(t)}_i\}$ be a set of graph-fact prompts and $a^{*(t)}_i$ the correct answers. The influence of head $(h, \ell)$ on task $t$:

$$\text{IE}^{(t)}(h, \ell) = \frac{1}{N}\sum_{i=1}^{N}\!\left[p_\theta(a^{*(t)}_i | x^{(t)}_i, \text{patch}_{h,\ell}) - p_\theta(a^{*(t)}_i | x^{(t)}_{\text{corrupt},i})\right]$$

The response map predicts the influence of a downstream head from an upstream edit:

$$\hat{\Delta}a_{h',\ell'} = R^{(t)}_{(h,\ell) \to (h',\ell')} \cdot \Delta a_{h,\ell}$$

The predictive quality is measured by mean squared error (MSE) relative to a naive zero predictor:

$$\text{MSE\_reduction} = 1 - \frac{\|{\Delta a_{h',\ell'}} - \hat{\Delta}a_{h',\ell'}\|^2}{\|\Delta a_{h',\ell'}\|^2}$$

---

## Learning or Inference Procedure

1. Construct graph-fact prompts for each of five task types (factual recall, translation, analogy, sentence completion, question answering).
2. For each model and task, run activation patching to compute $\text{IE}^{(t)}(h, \ell)$ for all heads.
3. Fit response maps $R^{(t)}$ on 80% of prompts; evaluate MSE reduction on the held-out 20%.
4. Perform activation edits at identified early heads and measure downstream answer-token probability changes.
5. Compare influence profiles across tasks with a cosine-similarity matrix over the influence vectors.

---

## What the Paper Claims

- Task-switching redistributes influence across heads even when the factual content is identical.
- Response maps trained on one task transfer to a different task on the same factual content, with 50–70% of the MSE reduction preserved.
- Early activation edits (at layer 3–7 of 28) can redirect which entity the model retrieves without editing the instruction prefix.
- In Gemma-2-9B, a single targeted edit reduces factual-answer MSE by 53.6%; in Qwen3-14B by 5.0% (more distributed computation in the larger model).

---

## Experimental Findings

- **Models:** Phi-3-mini-3.8B, Gemma-2-9B, Llama-3.1-8B, Qwen3-14B, Mistral-7B.
- **Tasks:** Factual recall, translation, analogy, sentence completion, open-ended QA.
- **Key result:** Cosine similarity between influence profiles across tasks averages 0.41 (range 0.18–0.67), indicating substantial but incomplete overlap.
- **Response map performance:** Gemma-2-9B shows 53.6% MSE reduction; Llama-3.1-8B 41.2%; Phi-3-mini 38.8%; Qwen3-14B 5.0%.
- **Cross-task transfer:** Maps trained on factual recall transfer best to sentence completion (0.62 similarity) and worst to translation (0.19 similarity).

---

## Ablations and Interpretation

- **Fixed-task vs. varying-task maps:** Response maps fitted on a single task outperform cross-task maps by 15–20% MSE on that task, confirming task-specific routing.
- **Layer-range specificity:** Response maps from mid-layer edits (layers 10–20) are less predictive than early-layer maps, suggesting early layers are the primary routing bottleneck.
- **Attention vs. MLP contribution:** Attention heads account for 70–85% of the task-dependent influence redistribution; MLP contributions are less variable across tasks.
- **Implications for alignment:** A circuit that appears "safe" on task A may be activated under task B with no corresponding safety intervention; cross-task circuit analysis is recommended before deployment.

---

## Reference

Sovesh Mohapatra, Dani S Bassett. **Task Context Shapes Component Roles in Language Models**. arXiv:2610.13895.  
https://arxiv.org/abs/2610.13895
