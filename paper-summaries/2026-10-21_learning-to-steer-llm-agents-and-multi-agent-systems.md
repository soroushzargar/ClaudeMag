# Learning to Steer LLM Agents and Multi-Agent Systems

**arXiv:** 2610.22316  
**Venue:** arXiv preprint, October 2026  
**Authors:** Leyang Shen, Yaxin Gong, et al.  
**Date:** October 15, 2026

---

## One-Line Summary

A dedicated steering agent with dual intervention paths (fast/slow), jointly trained with GRPO, reduces goal drift in both single-agent and multi-agent LLM systems and transfers to unseen executors and tasks.

---

## Problem

LLM agents and multi-agent systems are prone to goal drift: as execution unfolds step by step, the system progressively deviates from the original high-level objective. Individual agents optimize locally, communication overhead between agents compounds errors, and no component has global oversight of whether the execution trajectory is on course. The result is wasted computation on unproductive sub-tasks and, in the worst case, agents that complete the literal instruction but miss the intended goal.

---

## Key Observation: A Separate Overseer Can Correct without Stopping Execution

Rather than redesigning every agent to reason globally, the authors introduce a dedicated steering agent that sits outside the executor(s). It observes the execution context at each step and decides both *whether* to intervene and *how* to intervene, without halting the underlying task.

---

## Why Existing Approaches Fall Short

- **Prompt-based correction:** Injecting correction instructions into the next step is brittle; it lacks a trained policy for choosing when and how to intervene.
- **Reward shaping in training:** Re-training all agents with a global objective is expensive and does not compose easily when agents are heterogeneous or off-the-shelf.
- **Post-hoc replanning:** Replanning from scratch after detecting failure is costly and cannot prevent drift mid-execution.

---

## Core Method

**Dual intervention paths:**
- **Fast path (low-cost):** Handles common drift patterns with lightweight corrective actions (e.g., re-anchoring the agent to the current sub-goal, pruning off-track tool calls).
- **Slow path (high-cost):** Invoked for complex or novel deviations; generates a full re-planning step or a coordinator message to all agents.

The steering agent decides which path to invoke (or to stay silent) based on the execution context. Both paths and the routing decision are trained jointly.

**Training with GRPO and pairwise ranking:**
The steering agent is trained with Group Relative Policy Optimization (GRPO). A pairwise ranking objective rewards interventions that, in hindsight, produce better final outcomes than the counterfactual of no intervention or a different intervention type. This avoids dense reward engineering while still capturing what "useful steering" means.

---

## Technical Formulation

Let $s_t$ be the execution state at step $t$ and $a^{\text{exec}}_t$ the action the executor would take. The steering agent chooses $\delta_t \in \{\text{silent}, \text{fast}, \text{slow}\}$ and, if not silent, produces a steering signal $\sigma_t$.

The executor's effective action is then:

$$a_t = f(a^{\text{exec}}_t, \sigma_t)$$

where $f$ merges the executor's action with the steering signal (override, prepend, or augment, depending on the interface).

The GRPO objective for the steering agent is:

$$\mathcal{J}_{\text{steer}}(\theta) = \mathbb{E}\left[\sum_t \left(R_{\text{final}} - \lambda_{\text{cost}} \cdot \mathbf{1}[\delta_t = \text{slow}]\right) \cdot \log \pi_\theta(\delta_t, \sigma_t \mid s_t)\right]$$

with $\lambda_{\text{cost}}$ penalizing expensive slow-path invocations, encouraging the agent to use the fast path whenever sufficient.

The pairwise ranking loss supplements GRPO by comparing pairs of trajectories (same task, different steering decisions) and penalizing the policy that produced the lower-quality outcome.

---

## Learning or Inference Procedure

1. Collect paired trajectories: one with steering agent active, one without (or with a different steering decision).
2. Annotate outcomes and form pairwise preference labels.
3. Train steering agent with joint GRPO + pairwise ranking loss.
4. At inference: steering agent runs alongside the executor(s), intervening via fast or slow path as its policy dictates.
5. The executor agents are not modified; the interface is generic enough to support heterogeneous executors.

---

## What the Paper Claims

- **Performance:** The steering agent improves task completion on information-seeking benchmarks for both single-agent and multi-agent settings.
- **Efficiency:** Fewer total steps consumed per task due to earlier course-correction.
- **Transfer:** The steering policy transfers to out-of-domain tasks and unseen executors without any fine-tuning of the steer agent.
- **Controllable intervention frequency:** Steering frequency can be adjusted post-training via the $\lambda_{\text{cost}}$ parameter, enabling efficiency-accuracy trade-offs.

---

## Experimental Findings

- **Settings:** Single-agent information-seeking (web research, document QA) and multi-agent pipeline tasks.
- **Baselines:** Executor-only (no steering), prompt-injected correction, and replanning-on-failure.
- **Key results:** The steering agent consistently outperforms all baselines; gains are larger for multi-agent settings, where drift compounds across agents.
- **Transfer:** The same steering policy, without modification, generalizes to executors it never encountered during training.

---

## Ablations and Interpretation

- **Fast-only steering:** Loses performance on novel drift patterns that require deeper re-planning; the slow path is necessary.
- **Slow-only steering:** Higher cost per step; the fast path handles the majority of cases cheaply.
- **Without pairwise ranking (GRPO alone):** Intervention decisions are noisier; the agent more often invokes the slow path when the fast path would have sufficed.
- **Why it works:** The pairwise objective provides a clean contrastive signal: the model learns which drift patterns are cheaply correctable and which require deeper intervention.

---

## Reference

Leyang Shen, Yaxin Gong, et al. **Learning to Steer LLM Agents and Multi-Agent Systems**. arXiv:2610.22316. October 2026.  
https://arxiv.org/abs/2610.22316
