# Le Critique: Privileged Value Functions for LLM Reinforcement Learning

**arXiv:** 2608.16739  
**Authors:** Siddarth Venkatraman, Matthieu Dinot, Laurence Aitchison  
**Submitted:** August 17, 2026  
**Area:** LLM Reinforcement Learning, Credit Assignment, Value Functions

---

## Summary

Le Critique identifies a fundamental tension in LLM reinforcement learning between the efficiency of group-relative methods (GRPO) and the credit-assignment quality of value-function methods (PPO), and proposes two techniques that capture benefits of both: Privileged Value Functions (PVF), which give critics access to task-relevant information unavailable to the policy, and TETHER, an adaptive baseline that interpolates between group-relative and value-based advantages.

## Problem

Reinforcement learning for LLMs is dominated by two paradigms:

**Group-relative methods (GRPO, REINFORCE++):** Sample a group of $G$ rollouts per prompt, compute rewards, and use the mean (or other statistic) as a baseline. Gradient variance is low when $G$ is large, but:
- Credit is assigned only at sequence level — all tokens in a response receive the same advantage signal
- Training is bottlenecked by the slowest rollout in the group ("straggler problem"), reducing effective throughput
- Off-policyness accumulates as the policy drifts during group completion

**Value-function methods (PPO):** Maintain a learned critic $V_\phi(s_t)$ that predicts return from state $s_t$, enabling token-level advantages. But:
- Training a reliable critic from the policy's own token context is hard: the policy's representations evolve during training, making critic targets non-stationary
- A separate critic doubles memory and compute

The core difficulty is that the policy's token context contains *everything the policy knows*, but for many tasks the optimal value function would use *more* information — such as the ground-truth answer, the rubric, or the current step count — that the policy does not have access to during inference.

## Method

**Privileged Value Functions (PVF):** The critic $V_\phi(s_t, p_t)$ receives both the standard policy context $s_t$ (the token sequence so far) and a privileged context $p_t$ (additional task-relevant information available in the environment but not to the policy at inference time). This privileged context might include:
- The ground-truth answer (for math reasoning)
- The oracle plan or target trajectory (for agentic tasks)
- Step-level binary correctness signals (for code generation)

The policy is trained with token-level advantages $\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}, p_{t+1}) - V_\phi(s_t, p_t)$, using gradients that flow through the policy head only, not the critic.

At inference time, the critic is discarded entirely — privileged information is never accessible to the policy.

**TETHER:** An adaptive baseline that interpolates between the group-relative baseline $b_\text{GR}$ and the value-function baseline $b_V$ based on estimated critic reliability $\hat{\rho}$:

$$b_t = \hat{\rho}_t \cdot V_\phi(s_t, p_t) + (1 - \hat{\rho}_t) \cdot b_\text{GR}$$

When the critic is well-calibrated ($\hat{\rho} \approx 1$), TETHER uses token-level advantages. When the critic is unreliable ($\hat{\rho} \approx 0$), it falls back to group-relative baselines. Critic reliability is estimated online from the critic's loss on held-out rollouts.

## Technical Formulation

Let $\mathcal{T} = (s_0, a_0, r_0, \ldots, s_T)$ be a trajectory. The policy gradient objective with TETHER is:

$$\nabla_\theta J = \mathbb{E}_\mathcal{T}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t^\text{TETHER}\right]$$

where:
$$\hat{A}_t^\text{TETHER} = r_t + \gamma b_{t+1} - b_t$$
$$b_t = \hat{\rho} \cdot V_\phi(s_t, p_t) + (1 - \hat{\rho}) \cdot \mathbb{E}_G[R(\mathcal{T})]$$

Critic reliability:
$$\hat{\rho} = \sigma\left(-\alpha \cdot \mathcal{L}_V^\text{hold-out}\right)$$
where $\sigma$ is sigmoid, $\alpha$ is a temperature, and $\mathcal{L}_V^\text{hold-out}$ is the critic loss on a held-out 10% of rollouts.

## Experimental Results

- PVF with ground-truth privileged context improves RLVR on GSM8K and MATH from GRPO baseline by **+4.2 accuracy points** with the same training budget
- TETHER with PVF achieves better sample efficiency than PPO, reaching GRPO final accuracy **2.3× faster** in training steps
- TETHER reduces the straggler bottleneck by enabling asynchronous rollout completion: when one rollout finishes, the adaptive baseline can update without waiting for the rest of the group
- Ablations confirm that privileged information is critical: PVF without privileged context underperforms GRPO

## Key Contributions

1. A principled framework for giving critics access to information beyond the policy's token context, with theoretical justification for why this improves credit assignment
2. An adaptive interpolation baseline (TETHER) that gracefully degrades to group-relative methods when critics are unreliable
3. Empirical demonstration that straggler bottlenecks in group-relative training are a real throughput limiter, addressable through value-function parallelism

## Significance

Le Critique points toward a future where critics in LLM RL are specialized agents with their own information channels, rather than copies of the policy trying to estimate value from the same context. The privileged information paradigm connects LLM RL with the robotics literature on privileged learning, where teachers with oracle information help train students that must operate without it.
