# SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning

**Authors:** Jinyang Wu, Shuo Yang, Zhengxi Lu, Fan Zhang, Yuhao Shen, Lang Feng, Haoran Luo, Zheng Lian, Shuai Zhang, Zhengqi Wen, Jianhua Tao  
**Affiliation:** University of Science and Technology of China  
**Venue:** arXiv:2607.14777  
**Date:** July 16, 2026  
**URL:** https://arxiv.org/abs/2607.14777

---

## Summary

Long-horizon agentic tasks—web navigation, interactive tool use, multi-step code execution—require LLMs to make decisions over many steps where sparse outcome rewards provide little token-level guidance. SEED converts completed on-policy trajectories into natural-language hindsight skills and distills them back into the policy in a self-evolving loop, where the same model checkpoint simultaneously acts in the environment, analyzes its own trajectories, and receives distilled supervision, with the cycle repeating as the policy improves.

## Problem

Reinforcement learning for long-horizon agentic tasks provides outcome-level rewards: the agent receives a signal only at the end of a multi-step trajectory, which may span 20–50 decision steps. Token-level policy gradients must propagate through this credit assignment gap.

Existing approaches:
- **Pure outcome RL:** Gradient signal is sparse and noisy; the policy may learn correct final actions but wrong intermediate strategies.
- **Static distillation:** Distill a stronger teacher into the student, but the teacher is fixed and may not cover the current policy's error modes.
- **Off-policy hindsight:** Learn from past successful trajectories, but trajectories generated under earlier policies may be out-of-distribution for the current policy.

## Core Idea: Self-Evolving Distillation

SEED addresses these gaps with a co-evolutionary loop:

1. **Act:** The current policy $\pi_k$ interacts with the environment, producing a batch of on-policy trajectories $\{\tau_i\}$.
2. **Analyze:** The same policy $\pi_k$ reads each completed trajectory and generates a natural-language **hindsight skill** $s_i$—a concise description of the reusable decision patterns, crucial observations, and failure modes in that trajectory.
3. **Distill:** The policy is fine-tuned on $(x_i, s_i, a_i)$ triplets—problem context, hindsight skill, correct action—using a behavior cloning objective.
4. **Evolve:** The updated policy $\pi_{k+1}$ becomes the next actor and analyzer. The hindsight skills in the next round reflect the policy's new capabilities and remaining failure modes.

Because the analyzer and the policy share weights, hindsight skills are automatically aligned with the current policy's behavior: they highlight what the current policy tends to get right or wrong, not what a fixed external model would diagnose.

## Technical Formulation

Let $\tau = (o_1, a_1, o_2, a_2, \ldots, o_T, a_T, R)$ be a trajectory with observations $o_t$, actions $a_t$, and outcome reward $R$. The hindsight skill generation step solves:

$$s_i = \pi_k\!\left(\cdot \mid \text{Analyze}(\tau_i)\right)$$

where $\text{Analyze}(\tau)$ is a prompt instructing the model to identify: (a) which intermediate decisions contributed most to the outcome, (b) which observations were decisive, and (c) which alternative actions would have improved the result.

The distillation step optimizes:

$$\mathcal{L}_\text{SEED}(\theta) = -\mathbb{E}_{(x,s,a) \sim \mathcal{D}_k} \left[\log \pi_\theta(a \mid x, s)\right]$$

where $\mathcal{D}_k$ is the dataset of $(x_i, s_i, a_i)$ triplets collected by $\pi_k$.

The full SEED update alternates between outcome RL (using GRPO on trajectory-level rewards) and distillation (using $\mathcal{L}_\text{SEED}$), with mixing ratio $\alpha$:

$$\mathcal{L}_\text{total} = (1 - \alpha)\mathcal{L}_\text{GRPO} + \alpha \mathcal{L}_\text{SEED}$$

## Key Results

Experiments on AlfWorld (household task completion) and WebArena (web navigation):

**AlfWorld (success rate ↑):**
| Method | 3B | 7B |
|--------|----|----|
| GRPO only | 61.3 | 72.8 |
| Static distillation | 67.4 | 78.1 |
| SEED (ours) | **74.2** | **83.5** |

**WebArena (task success ↑):**
| Method | 7B |
|--------|----|
| GRPO only | 18.4 |
| Static distillation | 21.7 |
| SEED (ours) | **26.3** |

SEED consistently outperforms both pure RL and static distillation baselines, with larger gains at smaller model sizes (3B)—suggesting that hindsight skill supervision is particularly valuable when model capacity is limited.

## Ablation: Self-Evolution Matters

- **Fixed analyzer (no evolution):** Accuracy drops by 5.1pp on AlfWorld, confirming that stale analyzers produce out-of-distribution skills.
- **Random skills (scrambled $s_i$):** Accuracy drops to below GRPO only, confirming that hindsight content (not just fine-tuning format) is responsible for improvements.
- **Analysis only, no distillation:** Accuracy improves by 1.4pp over GRPO-only, suggesting that the act of analyzing trajectories provides some benefit even without formal distillation.

## Significance

SEED introduces self-supervised hindsight generation as a bridge between sparse outcome rewards and dense token-level supervision in agentic RL. The self-evolving loop ensures that supervision stays aligned with the current policy throughout training—a property that static distillation cannot provide.
