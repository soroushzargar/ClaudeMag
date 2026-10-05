# The Missing Primitive: Diagnosing and Repairing Mathematical Reasoning in Large Language Models

**arXiv:** 2610.02191  
**Authors:** Shuo Xing, Zilin Dai, Chengyuan Qian, Fangzhou Lin, Wenjing Chen, Ping He, Pan Lu, Alvaro Velasquez, Mohit Bansal, Zhengzhong Tu  
**Affiliation:** Texas A&M University; UNC Chapel Hill; UCLA; University of Colorado Denver  
**Date:** October 1, 2026

---

## One-Line Summary

By introducing a new concept called Mathematical Primitive and a four-dimensional benchmark called Prim, this paper shows that independent primitive discovery—not execution—is the dominant bottleneck to LLM mathematical reasoning, and that primitive-guided post-training yields consistent improvements.

---

## Problem

LLMs achieve striking performance on frontier mathematics benchmarks such as AIME and MATH, yet it remains unclear whether these models possess genuine structural mathematical understanding or whether they are pattern-matching on surface features. A model might solve a problem through brittle memorization or procedural heuristics without grasping the foundational idea organizing the solution. This raises two questions: (1) Can we precisely diagnose the mathematical capabilities underlying a given solving accuracy? (2) Can we leverage those findings to improve post-training?

---

## Why Existing Approaches Fall Short

- **Benchmark saturation:** State-of-the-art LLMs now score highly on MATH, AMC, and even AIME, but high scores mask vastly different capability profiles—models with similar final accuracy may rely on completely different mechanisms.
- **No structural audit:** Existing benchmarks measure whether a model produces a correct final answer, not whether it has identified the core organizational idea underlying the solution. Two models with identical answer accuracy may differ dramatically in their structural understanding.
- **Post-training targeting:** Standard chain-of-thought fine-tuning and process reward models treat all reasoning steps equally; they do not specifically target the discovery of foundational mathematical structures where the diagnosis reveals the greatest weakness.

---

## Core Method

**Mathematical Primitive.** The paper defines a Mathematical Primitive as a concise, non-procedural representation of the foundational idea that organizes a solution—essentially the insight that, once identified, makes the detailed execution steps straightforward. For example, the primitive for a combinatorics problem might be "map to a stars-and-bars counting argument" rather than any particular arithmetic sequence.

**Prim Benchmark.** The authors construct Prim, a benchmark spanning four complementary capability dimensions:
- **Discovery:** Given a problem statement, identify the correct primitive without step-by-step scaffolding.
- **Generation:** Given a primitive, produce a valid solution execution.
- **Digestion:** Given a complete solution, extract the primitive it relies on.
- **Execution:** Given both a problem and its primitive, carry out the detailed solution steps.

These four dimensions are carefully designed so that performance on each is interpretable on its own and collectively triangulates the model's structural understanding.

---

## Technical Formulation

Let $\mathcal{P}$ be the space of mathematical primitives and $\mathcal{Q}$ be the space of problem instances. A problem $q \in \mathcal{Q}$ has a ground-truth primitive $p^*(q) \in \mathcal{P}$ and a correct solution $s^*(q)$.

Define four capability functions:
$$\text{Disc}(q) = \mathbf{1}[\hat{p}(q) = p^*(q)], \quad \text{Gen}(p^*(q)) = \mathbf{1}[\hat{s}(p^*(q)) \text{ is correct}]$$
$$\text{Dig}(s^*(q)) = \mathbf{1}[\hat{p}(s^*(q)) = p^*(q)], \quad \text{Exec}(q, p^*(q)) = \mathbf{1}[\hat{s}(q, p^*(q)) \text{ is correct}]$$

The key empirical finding is that $\text{Exec}(q, p^*(q)) \gg \text{Disc}(q)$ for all tested models, meaning the bottleneck is Discovery, not Execution: given the correct primitive, models can execute the solution much more reliably than they can independently discover the primitive.

---

## Learning or Inference Procedure

**Primitive-guided post-training:**
1. For each training problem, the authors either annotate or LLM-generate candidate primitives.
2. A primitive discovery objective is added to the standard next-token prediction or RLVR loss, weighting gradient updates toward steps involving primitive identification.
3. A primitive-conditional generation objective trains the model to produce correct solutions conditioned on provided primitives.

**Evaluation procedure:**
1. Run each of the four Prim dimensions separately.
2. Compute per-model capability profiles across Discovery, Generation, Digestion, and Execution.
3. Compare capability profiles across models with similar final task accuracy to reveal hidden differences in structural understanding.

---

## What the Guarantee Says

The paper does not provide a formal convergence guarantee. The central claim is an empirical diagnostic finding: across all tested models (covering frontier-scale LLMs), the Execution score given the correct primitive substantially exceeds the unconditioned Discovery score, establishing Discovery as the universally dominant bottleneck. The post-training improvement is demonstrated empirically across multiple LLM families.

---

## Experimental Findings

- **Discovery bottleneck universality:** Across all tested LLMs, Execution given correct primitive exceeds unconditioned Discovery by a large margin—in some models by more than 30 percentage points.
- **Hidden capability differences:** Two models with identical MATH accuracy can differ by over 20 percentage points on the Discovery dimension alone, revealing that leaderboard accuracy is an insufficient proxy for structural understanding.
- **Latent execution capacity:** Providing correct primitives to models that cannot independently discover them unlocks near-full solution accuracy, confirming that execution chains are largely available; the bottleneck is the initial structural insight.
- **Post-training improvement:** Primitive-guided post-training consistently improves both Discovery accuracy and overall task performance, with improvements transferring to held-out datasets not used during training.

---

## Ablations and Interpretation

- **Primitive granularity:** Coarser primitives (e.g., "use combinatorics") yield lower Execution improvement than fine-grained ones ("map to stars-and-bars"), confirming that precision of the primitive matters.
- **Primitive quality at inference:** Using LLM-generated rather than ground-truth primitives during evaluation reduces but does not eliminate the Execution advantage, suggesting even imperfect discovery guidance is beneficial.
- **Digestion vs. Discovery correlation:** High Digestion accuracy does not predict high Discovery accuracy—models can recognize a primitive when shown a solution but cannot independently identify it, indicating that recognition and generation are distinct capabilities.

---

## Reference

Shuo Xing, Zilin Dai, Chengyuan Qian, Fangzhou Lin, Wenjing Chen, Ping He, Pan Lu, Alvaro Velasquez, Mohit Bansal, Zhengzhong Tu. **The Missing Primitive: Diagnosing and Repairing Mathematical Reasoning in Large Language Models**. arXiv:2610.02191, October 2026.  
https://arxiv.org/abs/2610.02191
