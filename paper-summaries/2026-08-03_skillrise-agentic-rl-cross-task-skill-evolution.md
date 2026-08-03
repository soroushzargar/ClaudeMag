# SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution

**Authors:** Zhiyuan Yao, Yuxin Chen, Zhengxi Lu, Zishan Xu, Yueqing Sun, Yifu Guo, Yuquan Lu, Zhengzhou Cai, Kangning Zhang, Zhuowen Han, Zihan Wang, Ziang Ye, Qi Gu, Xunliang Cai, Weiwen Liu, Yongliang Shen  
**Affiliation:** Zhejiang University; National University of Singapore; Shanghai Jiao Tong University; Meituan  
**Venue:** arXiv:2607.26784  
**Date:** July 29, 2026  
**URL:** https://arxiv.org/abs/2607.26784

---

## Summary

Standard agentic reinforcement learning trains an LLM-based agent on each task independently, discarding any generalizable patterns learned across episodes. SkillRise teaches the agent two roles at once: task solver and skill curator. After each task attempt, the same policy generates an update to a natural-language skill document that captures reusable workflows. This document is passed as context to the next task. A decoupled credit assignment scheme ensures that skill curation is rewarded for its downstream impact—not just for producing plausible-looking documentation. On ALFWorld, WebShop, and ScienceWorld, SkillRise achieves the best Pass@1 among compared methods.

## Problem

Agentic RL for LLMs (RLVR over web navigation, embodied tasks, scientific workflows) has two recurring issues:

1. **No cross-task transfer.** Each new task starts from scratch. Successful action sequences from similar tasks in the training set are discarded after training, except insofar as the policy weights encode them.
2. **Skill extraction is pipeline-heavy.** Prior work (Voyager, SKILL1, SkillRL) separates skill extraction from execution into multi-stage pipelines: extract skills from completed trajectories, store them in a retrieval database, retrieve them for future tasks. This creates brittle interfaces between stages and requires separate training objectives for each component.

SkillRise addresses both by jointly training a single policy to perform task solving and skill curation within one RL objective.

## Method: Unified Policy with Skill Documents

A skill document $S_t$ is a natural language description of reusable workflows accumulated across the first $t$ tasks. The policy $\pi_\theta$ is given two modes of operation:

1. **Solve mode:** Given task $T_t$ and skill document $S_{t-1}$, generate a trajectory of actions $a_{1:H}$ to solve the task.
2. **Curate mode:** After completing $T_t$, given the trajectory $a_{1:H}$ and current document $S_{t-1}$, generate an updated skill document $S_t$ by adding, revising, or deleting workflow entries.

Both modes use the same model weights $\theta$. The sequence of tasks $T_1, T_2, \ldots, T_N$ is organized into progressively challenging instances (easier instances first), so that the skill document grows in informativeness as the agent encounters more difficult variations.

## Technical Formulation

Let $r_{\text{solve}}(T_t, a_{1:H})$ be the task completion reward for task $T_t$. Let $r_{\text{curate}}(S_t, T_{t+1}, \ldots, T_{t+k})$ be the discounted downstream reward measuring how much the updated skill document $S_t$ improves performance on the next $k$ tasks.

The SkillRise objective is:

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[\sum_{t=1}^{N} r_{\text{solve}}(T_t, a_{1:H}^{(t)}) + \gamma \cdot r_{\text{curate}}(S_t, T_{t+1:t+k})\right]$$

where $\gamma \in (0,1)$ is a discount factor that down-weights credit from tasks far in the future. This is the key innovation: the curation reward is not assigned at the time of skill generation but is computed from the actual impact on future task completion.

In practice, $r_{\text{curate}}$ is approximated by:

$$r_{\text{curate}}(S_t, T_{t+1:t+k}) = \frac{1}{k} \sum_{j=1}^{k} \gamma^j \cdot \Delta r_{\text{solve}}(T_{t+j}, S_t \text{ vs. } S_{t-1})$$

where $\Delta r_{\text{solve}}$ measures the improvement in task completion rate when $S_t$ rather than $S_{t-1}$ is used as context for task $T_{t+j}$.

## Learning Procedure

Training uses Group Relative Policy Optimization (GRPO) rather than PPO, sampling multiple curation outputs for each post-task state and comparing them by their downstream impact. The training curriculum:

1. Tasks are grouped by type and ordered from simple to complex.
2. Within each group, the agent performs solve-curate cycles, with the skill document updated after each task.
3. GRPO updates the policy based on the relative advantage between different curation outputs, encouraging the policy to produce skill documents that maximize downstream pass rates.

A skill document is constrained to at most 2,000 tokens to fit within the context window alongside task instructions.

## What the Guarantee Says

The decoupled credit assignment provides a monotonicity property: if skill curation were perfect (the document always improves future tasks), the SkillRise objective reduces to the standard RLVR objective plus a term proportional to the number of future tasks benefited. Empirically, this means the agent has an incentive to produce informative skills even early in training, when individual tasks may be too hard to solve from scratch.

## Experimental Findings

**ALFWorld (interactive text-based household tasks):**

| Method | Pass@1 (zero-shot) |
|---|---|
| Base LLM (no RL) | 42.3% |
| Standard RLVR | 67.8% |
| SKILL1 | 71.4% |
| SkillRise (ours) | **78.9%** |

**WebShop (web-based e-commerce navigation):**

| Method | Task Success |
|---|---|
| Standard RLVR | 56.2% |
| SkillFlow | 61.4% |
| SkillRise (ours) | **66.3%** |

**ScienceWorld (scientific experiment tasks):**

| Method | Pass@1 |
|---|---|
| Standard RLVR | 38.1% |
| SKILLC | 43.7% |
| SkillRise (ours) | **50.2%** |

Gains over standard RLVR range from 11 to 12 percentage points, confirming that joint solve-curate training with decoupled credit assignment yields substantial improvements over both no-skill and separate-skill-extraction baselines.

## Ablations and Interpretation

**Decoupled credit assignment:** Replacing $r_{\text{curate}}$ with an immediate reward (e.g., document fluency or coverage) reduces ALFWorld performance to 72.3%, confirming that downstream impact is the critical signal.

**Shared vs. separate policy:** Training a separate curation policy (distinct weights) reduces Pass@1 by 4.1 points on ALFWorld, suggesting that joint training creates productive interference: solving experience improves curation quality and vice versa.

**Skill document length:** Allowing up to 4,000 tokens provides a 1.2-point gain over 2,000 tokens on ScienceWorld, suggesting the document length constraint is occasionally binding for complex domains.

**Curriculum ordering:** Random task ordering (no progression from easy to hard) reduces ALFWorld Pass@1 by 6.3 points, confirming that the curriculum is important for effective early skill accumulation.
