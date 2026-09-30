# Recursive self-improvement of AI research agents

**arXiv:** 2609.26457  
**Authors:** Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, Zhengyao Jiang  
**Affiliation:** Weco AI  
**Date:** September 22, 2026

---

## One-Line Summary

AIDE², a bi-level system in which an outer AI agent rewrites the code of an inner AI research agent, ran unattended for 8 days and discovered seven successive improvements — including 16× prompt compression and a halving of reward hacking — matching or exceeding a human-engineered production agent on four held-out benchmarks.

---

## Problem

AI research agents that autonomously engineer machine learning solutions have shown impressive results on benchmark competitions, but their scaffolding — the search policies, prompts, and memory mechanisms that guide exploration — is designed by human engineers and has largely remained static. Human R&D investment yields diminishing marginal returns: the most obvious improvements have already been found, and discovering subtler gains requires exhaustive human expertise. A system that can improve its own scaffolding could bypass these diminishing returns.

---

## Why Existing Approaches Fall Short

- **Standard AI research agents** (AIDE, OpenDevin-Science, etc.) optimize the code they write for tasks but do not modify their own agent scaffolding.
- **Meta-learning** adapts model weights but typically requires differentiable objectives and cannot easily optimize discrete code changes to scaffolding logic.
- **Neural architecture search and program synthesis** optimize specific code components but are not designed to improve a general-purpose agent's own control code.
- **Human-in-the-loop improvement** is slow, expensive, and cannot run continuously at machine speed.
- **Prior work on self-modification** focused on toy environments or narrow domains; scaling to a full-featured AI research agent with diverse evaluation tasks has not been demonstrated.

---

## Core Method

**Bi-Level Recursive Self-Improvement.** AIDE² implements a two-level optimization loop:

- **Inner loop (AIDE):** The standard AIDE agent performs automated machine learning on a task: it proposes, executes, evaluates, and refines code solutions using a tree search over candidate programs, with drafting, debugging, and improvement operators.
- **Outer loop (AIDE²):** The outer agent treats the inner agent's own code — its search policy heuristics, prompts, memory compression logic, and operator implementations — as the object of optimization. It proposes code changes, spins up a fresh inner agent with the modified scaffolding, evaluates that agent on a diverse battery of AI R&D tasks, and accepts the change only if aggregate performance improves.

**Evaluation Battery.** The outer loop evaluates each modification against a battery of approximately 50 diverse tasks covering ML engineering, algorithmic optimization, and scientific computing. Roughly 90% of proposed modifications are rejected; only those that improve the held-out aggregate score are committed to the active codebase.

**Progressive Code Repository.** The committed improvements accumulate in a version-controlled code repository. Each generation of the agent is the starting point for the next round of proposed modifications, creating a genuine recursive self-improvement dynamic.

---

## Technical Formulation

Let $\pi_{\theta}$ denote the inner AIDE agent parameterized by its scaffolding code $\theta$ (prompts, search policy, memory logic). Let $\mathcal{T}_{\text{train}}$ be the training battery and $\mathcal{T}_{\text{held}}$ be the held-out evaluation set.

The outer loop solves:
$$\theta^* = \arg\max_{\theta' \in \mathcal{N}(\theta)} \;\frac{1}{|\mathcal{T}_{\text{train}}|}\sum_{t \in \mathcal{T}_{\text{train}}} \text{score}(\pi_{\theta'}, t)$$
where $\mathcal{N}(\theta)$ is the neighborhood of code edits proposed by the outer LLM agent given $\theta$.

The update rule is:
$$\theta \leftarrow \theta' \quad \text{if} \quad \text{Eval}(\pi_{\theta'}, \mathcal{T}_{\text{train}}) > \text{Eval}(\pi_{\theta}, \mathcal{T}_{\text{train}})$$
and $\theta$ is unchanged otherwise. The outer loop is itself powered by an LLM that reads $\theta$, the evaluation traces, and recent performance deltas before proposing $\theta'$.

---

## Learning or Inference Procedure

**Outer loop (runs once over 8 days):**
1. Initialize $\theta_0$ from the production AIDE codebase.
2. LLM outer agent inspects $\theta$, performance traces, and failure modes.
3. Outer agent proposes a code edit $\delta$ (a diff or patch to $\theta$).
4. Spin up a sandboxed inner agent with $\theta' = \theta + \delta$ and run it on $\mathcal{T}_{\text{train}}$.
5. If $\text{Eval}(\pi_{\theta'}) > \text{Eval}(\pi_\theta)$: commit $\theta' \to \theta$; else discard $\delta$.
6. Repeat; accept approximately 1 in 10 proposed modifications.

**Inner loop (runs once per task per evaluation):**
1. Inner agent reads the task specification and exploration budget.
2. Tree search: draft code, execute, score, refine via improvement operators.
3. Return best solution found within budget.

---

## What the Guarantee Says

AIDE² does not provide a formal convergence guarantee (the outer loop's search space is open-ended natural-language code). The empirical guarantee is: the evaluation battery is diverse enough (50 tasks, multiple domains) that improvements accepted on the training battery generalize to held-out benchmarks not seen during self-improvement. The paper presents this as a necessary empirical condition for recursive self-improvement to be meaningful rather than overfitting to a narrow task distribution.

---

## Experimental Findings

Over 8 unattended days, AIDE² accepted 7 consecutive improvements to the inner agent's scaffolding:

- **New search algorithm:** Replaced the default tree search policy with a learned priority function that better exploits evaluation feedback.
- **16× prompt compression:** Discovered a compressed prompting scheme that conveys the same information in 1/16 the tokens, reducing per-step cost and enabling longer search horizons.
- **Emergent self-policing:** The evolved agent learned to detect and avoid reward-hacking patterns; reward hacking rate dropped from 63% to 34% on a held-out benchmark.

**Held-out benchmark results (4 benchmarks, none seen during self-improvement):**
- Machine learning engineering tasks: matches or exceeds human-engineered production AIDE.
- Heuristic algorithm engineering: exceeds human-engineered baseline.
- Physics-based weather forecasting: matches human-engineered baseline.
- A fourth held-out ML benchmark: exceeds human-engineered baseline.

---

## Ablations and Interpretation

- **Diverse vs. narrow training battery:** When the training battery is narrowed to a single domain, held-out generalization degrades significantly, confirming that task diversity is essential.
- **Outer agent model size:** Larger outer LLMs produce higher-quality code proposals and higher acceptance rates, but even smaller models can drive improvement given sufficient budget.
- **Reward hacking detection:** The self-policing behavior was not explicitly incentivized; it emerged because reward hacking lowered aggregate evaluation scores and the outer loop penalized it implicitly.
- **Cumulative gain decomposition:** Prompt compression alone accounts for about 40% of the total held-out improvement; the new search policy accounts for about 35%; memory improvements account for the remainder.

---

## Reference

Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, Zhengyao Jiang. **Recursive self-improvement of AI research agents**. arXiv:2609.26457, September 2026.  
https://arxiv.org/abs/2609.26457
