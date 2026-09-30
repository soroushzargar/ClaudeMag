# From Token Importance to Conditional Removability: Rethinking Visual Token Pruning in Multimodal Large Language Models

**arXiv:** 2609.26484  
**Authors:** Shengli He et al.  
**Affiliation:** Not fully specified  
**Date:** September 22, 2026

---

## One-Line Summary

Visual token pruning criteria that ignore representation depth and inter-token deletion dependencies systematically mis-score removability; CoRePrune fixes both failures with progressive depth-aware perturbation tracking and set-conditioned refinement, retaining 90.3% of dense performance with 128 tokens and 51% faster prefill.

---

## Problem

Multimodal large language models (MLLMs) process images as sequences of visual tokens (typically hundreds to thousands per image) interleaved with text tokens. Long visual token sequences dominate prefill computation. To reduce this cost, training-free pruning methods score each visual token by importance or redundancy and drop low-scoring ones before or during the forward pass. Despite widespread adoption, these methods often degrade performance unexpectedly at aggressive budgets, and no principled account of why they fail has been established.

---

## Why Existing Approaches Fall Short

- **Importance-based pruning** (attention score, gradient norm) scores tokens in isolation, ignoring how the representation of one token changes when others are removed.
- **Redundancy-based pruning** (pairwise similarity) assumes spatially similar tokens are interchangeable, but spatial similarity at layer $l$ does not imply similarity at layer $l'$ after further processing.
- **Static depth assignment:** Most methods apply pruning at a single layer (early or late), not accounting for the fact that removability changes as representations evolve through depth.
- **Set-agnostic scoring:** All existing methods score each candidate token independently of which other tokens are being removed, ignoring set-level interaction effects that change individual marginals.

---

## Core Method

**Conditional Removability.** The key insight is that token removability is not a fixed property — it is conditioned on two factors:
1. **Representation depth:** Removing a token at layer 5 has a different effect than removing it at layer 20, because the representations have been increasingly fused with context.
2. **Deletion set context:** Removing a token when 100 others have already been removed has a different effect than removing it from the full set, because the remaining tokens have already had to compensate for prior removals.

**CoRePrune: Two-Stage Training-Free Framework.**

*Stage 1 — Progressive Perturbation-Aware Visual Pruning (PPVP):* Rather than assigning pruning decisions at a single layer, PPVP tracks the downstream perturbation of each candidate removal as visual representations evolve through the transformer. At each layer block, it updates perturbation estimates by re-probing the representation distance between the full-token and pruned-token paths. Tokens whose removal causes growing perturbation over depth are promoted in priority; tokens whose perturbation shrinks are deprioritized.

*Stage 2 — Set-Conditioned Refinement (SCR):* After visual-text attention interaction (where visual and text tokens first mix), SCR re-evaluates each remaining candidate token under the current deletion set. Tokens that would now provide net benefit given what has already been removed (rescue candidates) are recovered; tokens whose marginal contribution has collapsed are added to the deletion set.

---

## Technical Formulation

Let $V = \{v_1, \ldots, v_N\}$ be the visual token sequence at layer 0. Let $f_l(V)$ denote the visual representations at layer $l$ after a forward pass of the MLLM with token set $V$.

**Perturbation at depth $l$ for deletion set $D$:**
$$P_l(D) = \left\|f_l(V \setminus D) - f_l(V)\right\|_F$$

**PPVP priority update:** For candidate token $v_i$, track $P_l(\{v_i\})$ across $l = l_1, l_2, \ldots$; prioritize $v_i$ for selection if $P_{l_2}(\{v_i\}) < P_{l_1}(\{v_i\})$ (perturbation shrinking with depth, indicating safe removal at $l_2$).

**SCR marginal contribution:** After visual-text interaction at layer $l^*$, define the rescue score for a candidate $v_j$ under current deletion set $D$:
$$r(v_j; D) = P_{l^*}(D) - P_{l^*}(D \cup \{v_j\})$$
Recover $v_j$ if $r(v_j; D) > 0$; add $v_j$ to $D$ if $r(v_j; D) \ll 0$.

---

## Learning or Inference Procedure

CoRePrune is **training-free** and runs at inference time:

1. **Forward pass to layer $l_1$:** Compute initial importance/perturbation scores.
2. **PPVP:** Continue forward pass to layer $l_2 > l_1$; update perturbation estimates; rank candidates.
3. **Tentative deletion:** Remove the lowest-priority tokens to meet the target budget.
4. **Continue to layer $l^*$** (visual-text interaction layer).
5. **SCR:** Re-evaluate the deletion set; rescue tokens with positive marginal contribution; finalize the deletion set.
6. **Continue forward pass** with the pruned token set to generate the output.

Total overhead: two additional perturbation probes per image during the prefill phase, which is negligible relative to the attention computation savings from the reduced token count.

---

## What the Guarantee Says

CoRePrune does not provide a formal approximation bound, but it gives a constructive argument: if perturbation at depth $l_2$ is smaller than at $l_1$, then the model has already compensated for the removal by $l_2$, making the token safe to remove. SCR provides a post-hoc correction that provably reduces the instantaneous perturbation under the current deletion set, giving a greedy local optimum guarantee.

---

## Experimental Findings

Evaluated across five MLLM backbones covering:
- Standard image understanding (LLaVA-1.5, InternVL2)
- High-resolution inputs (LLaVA-HR)
- Video (VideoLLM)

At a budget of **128 visual tokens**:
- CoRePrune retains **90.3%** of dense-model performance (vs. 83–86% for prior methods).
- Reduces aggregate prefill time by **51.0%** (vs. 45–48% for prior methods).

At more aggressive budgets (64 tokens), the advantage grows because set-conditioned effects become stronger when the deletion set is large.

**Task breakdown:** The largest gains over baselines are on tasks requiring global reasoning (VQA, video QA) where deletion-set interactions are strongest; on local patch tasks, improvement is smaller.

---

## Ablations and Interpretation

- **PPVP alone:** Removes the depth-static assumption and recovers about half the total gain.
- **SCR alone:** Removes the set-agnostic assumption and recovers the other half.
- **Joint (CoRePrune):** The two stages are complementary; their joint gain exceeds the sum of individual gains, suggesting non-linear interaction.
- **Depth of PPVP tracking:** Tracking across 3+ layer blocks is needed; fewer blocks converges to depth-static methods.
- **SCR timing:** Applying SCR before visual-text interaction (too early) or at the very last layer (too late) both degrade performance; the visual-text fusion layer is the sweet spot.

---

## Reference

Shengli He et al. **From Token Importance to Conditional Removability: Rethinking Visual Token Pruning in Multimodal Large Language Models**. arXiv:2609.26484, September 2026.  
https://arxiv.org/abs/2609.26484
