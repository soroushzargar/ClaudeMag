# When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion

**Authors:** Todd Zhou  
**Venue:** arXiv:2607.20543  
**Date:** July 2026  
**URL:** https://arxiv.org/abs/2607.20543

---

## Summary

Reinforcement learning with verifiable rewards (RLVR) has become the dominant recipe for improving reasoning in large language models. However, this paper uncovers a previously under-reported failure mode: RLVR can improve one-sample accuracy (pass@1) while simultaneously making the model *worse* under repeated sampling (pass@k for large k). The author terms this **pass@k inversion**.

## Problem

When evaluating LLM reasoning, pass@k measures the probability that at least one of k independent samples is correct. A model that is better in expectation should achieve higher pass@k for any k. However, RLVR-trained models exhibit an anomaly: after training, the policy solves fewer *distinct* problems than the base model at large k, even when pass@1 improves significantly.

This reveals that RLVR compresses the diversity of the model's output distribution—sharpening high-probability trajectories while eliminating rare-but-correct ones.

## Mechanism: Boundary Prompts

The failure concentrates on **boundary prompts**: inputs where the base model contains rare correct trajectories that are recoverable by sampling but appear too infrequently to be reliably observed in finite RLVR rollout groups. 

The author proposes a two-mode account:
- **High-frequency correct trajectories:** RLVR sees and reinforces these readily, improving pass@1.
- **Low-frequency correct trajectories (boundary prompts):** RLVR rarely observes correct rollouts, and with only incorrect examples in its rollout groups, applies policy updates that suppress even those rare correct paths.

This is an **absence-of-evidence failure**: the algorithm interprets "no correct rollout observed" as evidence that the correct answer is unlikely, and moves probability mass away from the rare but valid trajectory.

## Technical Formulation

Let the base policy be $\pi_\text{base}$ and the RLVR policy be $\pi_\text{RL}$. For a problem $x$ with correct answer set $\mathcal{A}(x)$, define:

$$\text{pass}@k(x, \pi) = 1 - \prod_{a \in \mathcal{A}(x)}\left(1 - \pi(a \mid x)\right)^k$$

Pass@k inversion occurs when there exists a set $\mathcal{X}_\text{inv}$ of problems such that:

$$\text{pass}@1(x, \pi_\text{RL}) > \text{pass}@1(x, \pi_\text{base}) \quad \forall x$$

but

$$\sum_{x \in \mathcal{X}_\text{inv}} \text{pass}@k(x, \pi_\text{RL}) < \sum_{x \in \mathcal{X}_\text{inv}} \text{pass}@k(x, \pi_\text{base})$$

for large k. On boundary prompts, the base model's rare correct trajectories satisfy $\pi_\text{base}(a \mid x) \ll 1$ but $\pi_\text{base}(a \mid x) > 0$. RLVR without correction drives $\pi_\text{RL}(a \mid x) \to 0$.

## Proposed Fix: Per-Problem Base Anchoring (PBA)

Per-Problem Base Anchoring (PBA) is a deliberately simple proof-of-concept fix. It:
1. Detects boundary prompts at inference or training time by measuring base model entropy.
2. For risky prompts, anchors the RLVR policy toward the base distribution via a KL penalty term:
   $$\mathcal{L}_\text{PBA} = \mathcal{L}_\text{RLVR} + \lambda \cdot D_\text{KL}(\pi_\text{base} \| \pi_\text{RL})$$
   applied selectively on boundary prompts.
3. Alternatively, injects frozen-base correct evidence by prepending successful base-model rollouts as few-shot examples before policy rollout groups are collected.

PBA partially recovers the diversity lost by standard RLVR without sacrificing pass@1 gains on non-boundary prompts.

## Experimental Findings

- Pass@k inversion is demonstrated across standard MATH and GSM8K benchmarks using Llama-3 and Qwen2.5 base models with GRPO-style training.
- After RLVR, pass@1 improves by 5–12 percentage points, while pass@16 *decreases* by 3–8 points on a subset of boundary problems.
- The boundary subset constitutes approximately 15–25% of test problems depending on the base model.
- PBA reduces the pass@k degradation on boundary prompts by approximately 60%, while preserving 95% of pass@1 gains.

## Significance

This paper issues a concrete warning against evaluating RLVR models solely at pass@1: diversity collapse is real and measurable. For deployment contexts requiring diverse generation (beam search, voting, best-of-N), RLVR without diversity-aware training may actively harm reliability.
