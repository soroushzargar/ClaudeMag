# Expanding LLM Reasoning

**arXiv:** 2610.05584  
**Venue:** NeurIPS 2026  
**Authors:** Rian Atri, Evan Luo  
**Date:** October 2026

---

## One-Line Summary

By asking *where* inside an existing chain-of-thought trace an extra continuation branch should start—rather than sampling more complete chains—a simple "always branch from the last step" rule beats uniform placement and rivals learned routers, cutting compute relative to self-consistency at matched accuracy.

---

## Problem

Chain-of-thought (CoT) reasoning improves LLM accuracy on hard tasks, but generating full independent chains is expensive. Two dominant strategies—self-consistency (sample $n$ complete chains and majority-vote) and best-of-$n$ (sample and rerank)—require generating the entire reasoning trace from scratch for each sample.

The authors ask a narrower question: if we have already generated one chain $\tau$, at which step $t^* \in \tau$ should we branch a continuation to maximally improve the probability of reaching the correct answer? This is the **chain expansion problem**.

---

## Key Observation: Expansion Utility

The authors define **expansion utility** at step $t$ as the expected improvement in correctness from restarting the chain at step $t$:

$$U(t) = \mathbb{E}_{c \sim p(\cdot | \tau_{:t})}[\mathbf{1}[a(c) = a^*]] - \mathbf{1}[a(\tau) = a^*]$$

A positive $U(t)$ means branching at step $t$ is expected to recover a failed chain. They compute $U(t)$ empirically for every step across nine models on six benchmarks (41 model × benchmark cells), creating the largest study of within-chain continuation utility to date.

---

## Why Existing Approaches Fall Short

- **Self-consistency:** Generates $n$ full chains from scratch; misses the information already embedded in the partial trace $\tau_{:t}$.
- **Best-of-n with reranker:** Also requires full chains and a trained reranker; the reranker only selects, it does not identify recovery points.
- **Learned step-level critics:** Require training a process reward model, which is expensive and brittle; the authors find a training-free baseline nearly matches.

---

## Core Method: Expansion Utility Analysis and Routing

**Step 1 — Utility measurement:** For each of 41 cells, the authors sample 64 chains, compute $U(t)$ at each step, and identify whether a positional class exists (early, middle, last).

**Step 2 — Baseline policies:** Three policies are evaluated:
- **Uniform:** branch at a uniformly random step.
- **Always-last:** branch from the last completed step $|\tau|$.
- **Learned router:** a lightweight classifier trained to predict $U(t) > 0$ from prefix embeddings.

**Step 3 — Held-out evaluation:** Each policy is evaluated by generating $k$ continuations from the selected branch point and majority-voting.

---

## Technical Formulation

Let $\tau = (s_1, s_2, \ldots, s_T)$ be a chain with answer $a(\tau)$, and let $\tau_{:t}$ denote the prefix up to step $t$. The expected correctness after branching at $t$ with $k$ continuations:

$$\text{Acc}(t, k) = \mathbb{E}_{c_1, \ldots, c_k \sim p(\cdot | \tau_{:t})}\!\left[\text{MajorityVote}(a(c_1), \ldots, a(c_k)) = a^*\right]$$

The expansion policy $\pi^*$ selects:

$$t^* = \arg\max_{t \in \{1,\ldots,T\}} \text{Acc}(t, k)$$

Since $\text{Acc}(t, k)$ is not known at inference time without additional sampling, the authors compare the always-last heuristic $t^* = T$ against oracle selection (upper bound) and uniform placement (baseline).

For the **learned router**, the input is the prefix embedding $\phi(\tau_{:t})$ from the model's hidden state, and the output is a binary label $\hat{y}(t) = \mathbf{1}[U(t) > 0]$. The router is a 2-layer MLP trained with binary cross-entropy.

---

## Learning or Inference Procedure

1. Generate one chain $\tau$ per problem instance.
2. Apply the branching policy (always-last, uniform, or learned router) to select $t^*$.
3. Sample $k = 4$ continuations from $\tau_{:t^*}$ in parallel.
4. Majority-vote over the $k$ answers.
5. Compute accuracy and compare total generated tokens against self-consistency at matched output budget.

---

## What the Paper Claims

- **Always-last beats uniform:** On 38/41 cells, always-last matches or exceeds uniform branching, confirming that later steps capture more context.
- **Router shows no consistent gain over always-last:** The learned router beats uniform but does not significantly surpass always-last on the held-out set, suggesting that positional signal dominates semantic features.
- **Compute efficiency:** On DeepSeek-R1-Distill-Qwen-14B / MATH-500, always-last expansion with $k = 4$ achieves the same accuracy as 4-sample self-consistency using 0.774× the generated tokens.

---

## Experimental Findings

- **Models:** DeepSeek-R1-Distill-Qwen-14B, Qwen3-14B, Gemma-2-9B, QwQ-32B, and five smaller models.
- **Benchmarks:** MATH-500, AIME 2024/2025, AMC 2023, GPQA-Diamond, LiveCodeBench.
- **Always-last vs. self-consistency:** Matches accuracy with 0.774× tokens on MATH-500; on AIME 2024, matches accuracy with 0.81× tokens.
- **Oracle upper bound:** Oracle selection outperforms always-last by 4.25 points on average, indicating that a better router could further improve results.
- **Cross-fitted generalization:** Step selections learned on one set of continuations transfer to disjoint held-out continuations with +4.25-point average gain over uniform.

---

## Ablations and Interpretation

- **Branch depth:** Branching from the last vs. second-to-last step gives similar results; the final step carries the most cumulative context and lowest remaining uncertainty.
- **Continuation length:** Shorter continuations (few-step endings) are slightly less accurate than full-chain continuations, suggesting the model benefits from regenerating the final answer token sequence.
- **Number of branches $k$:** Accuracy improves log-linearly with $k$ up to $k = 8$; diminishing returns beyond.
- **Router architecture:** A 2-layer MLP outperforms linear probes and 1-layer MLPs; larger architectures (4+ layers) overfit on the training split.
- **Why always-last works:** Empirical analysis of $U(t)$ shows a monotonically decreasing variance profile: later steps have lower variance in continuation correctness, making them more stable branching points even when they don't maximize expected accuracy.

---

## Reference

Rian Atri, Evan Luo. **Expanding LLM Reasoning**. arXiv:2610.05584. NeurIPS 2026.  
https://arxiv.org/abs/2610.05584
