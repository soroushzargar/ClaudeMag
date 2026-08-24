# SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning

**arXiv:** 2608.19842  
**Authors:** Dayang Liang, Lang Feng, Bo An, Yunlong Liu  
**Submitted:** August 20, 2026  
**Area:** Agentic Reinforcement Learning, LLM Post-Training

---

## Summary

SAPO is a memory-efficient and compute-efficient framework for training LLMs on long-horizon agentic tasks that eliminates the need for a separate critic model while providing fine-grained token-level credit assignment — benefits previously thought to require either large rollout groups or a dedicated critic network.

## Problem

Agentic reinforcement learning for LLMs — training on multi-turn tasks such as web navigation, code execution, and tool use — faces a fundamental trilemma:

1. **PPO** requires a separate critic model, roughly doubling memory footprint and adding 33%+ runtime overhead per iteration.
2. **GRPO** avoids a critic by sampling large rollout groups per prompt to compute group-relative advantages, but provides only sequence-level credit and training throughput is bottlenecked by the slowest rollout in each group.
3. Both methods struggle with long-horizon credit assignment — assigning rewards back to early actions in a multi-step trajectory.

## Method

SAPO exploits the autoregressive structure of LLMs to share policy and value functions within a single backbone, producing all estimates in a single forward pass without a separate model.

**Causal boundary sharing:** At each reasoning step, the LLM's residual stream at distinct token positions serves as the policy head input (the *action* boundary) and the value head input (the *state* boundary). Since the transformer's causal mask means earlier positions cannot attend to later ones, these boundaries are informationally separate despite sharing all parameters up to that point.

**Dual optimization:** SAPO independently optimizes two objectives over the shared backbone:
- **PPO objective:** Clipped policy gradient at action boundaries
- **Auxiliary SARSA objective:** On-policy temporal difference learning for the value heads, bootstrapping from the next state's value estimate

**Advantage estimation:** A trajectory-level generalized advantage estimator (GAE) combines lambda-weighted multi-step returns with batch normalization across the mini-batch to stabilize advantage estimates across variable-length trajectories.

## Technical Formulation

Let $s_t$ denote the state (observation history) and $a_t$ the action at step $t$. SAPO uses a single LLM $f_\theta$ with action head $\pi_\theta(a_t | s_t)$ and value head $V_\theta(s_t)$, both reading from the same residual stream at their respective causal boundaries.

The PPO loss clips the probability ratio:
$$\mathcal{L}_\text{PPO} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

The SARSA value loss:
$$\mathcal{L}_V = \mathbb{E}_t\left[(V_\theta(s_t) - r_t - \gamma V_{\theta_\text{old}}(s_{t+1}))^2\right]$$

The advantage estimate uses lambda-return with batch normalization:
$$\hat{A}_t = \frac{\delta^\lambda_t - \mu_B}{\sigma_B + \varepsilon}$$
where $\delta^\lambda_t$ is the lambda-return residual, and $\mu_B$, $\sigma_B$ are the batch mean and standard deviation.

## Experimental Results

Evaluated on ALFWorld (text-based household task completion) and WebShop (web navigation for e-commerce) with Qwen2.5-1.5B and Qwen2.5-7B:

- SAPO outperforms PPO by **+15.1 percentage points** average success rate across all settings
- SAPO outperforms GRPO by **+12.1 percentage points** average success rate
- Reduces per-iteration runtime by **33.2% vs. PPO** (no separate critic forward/backward pass)
- Eliminates the memory overhead of a separate critic model (~50% of baseline)
- Training is stable across all settings without reward hacking or collapse

## Key Contributions

1. A principled way to obtain policy *and* value estimates from a single LLM backbone using causal boundaries, with no additional parameters
2. A dual PPO+SARSA optimization procedure that exploits on-policy data from a single rollout
3. A batch-normalized GAE that stabilizes training on variable-length agentic trajectories
4. Empirical demonstration that single-rollout training can match or exceed multi-rollout group methods on standard agentic benchmarks

## Significance

SAPO addresses a practical barrier to deploying RL-trained agents: the compute and memory cost of critic models or large rollout groups. By achieving PPO-quality credit assignment in a single forward pass with one rollout per step, SAPO makes it feasible to scale agentic RL to environments requiring many sequential decisions.
