# Diffusion Policy with Behavioral Advantage Correction for Offline Reinforcement Learning

**arXiv:** 2608.02332
**Authors:** Botao Dong, Longyang Huang, Ning Pang, Hongtian Chen
**Venue:** arXiv preprint
**Date:** August 3, 2026
**URL:** https://arxiv.org/abs/2608.02332

---

## Summary

Offline reinforcement learning suffers from distribution shift between the logged behavioral data and the learned policy, causing erroneous Q-value estimation that misguides optimization. This paper proposes Behavioral Advantage Corrected Policy Evaluation (BAC-PE), which uses the Q-function of the behavior policy to correct the learned policy's Q-function, mitigating both pessimistic conservatism and overestimation bias simultaneously. Diffusion models represent both policies, enabling rich multimodal action distributions and accurate distribution matching. The method comes with theoretical convergence guarantees and outperforms prior offline RL methods on standard benchmarks.

## Problem

In offline RL, the agent learns entirely from a fixed dataset of (state, action, reward, next-state) tuples collected by some behavior policy $\mu$. The learned policy $\pi$ may assign high values to out-of-distribution (OOD) actions that never appear in the dataset. Standard Q-learning then propagates these erroneous estimates, leading to:

- **Overestimation:** The learned policy exploits actions with inflated Q-values, degrading real-world performance.
- **Pessimistic conservatism:** Penalty-based corrections (e.g., CQL) suppress Q-values so aggressively that even good near-distribution actions are undervalued.

Neither extreme is satisfactory: overestimation leads to unsafe policies, while excessive conservatism yields overly cautious policies that fail to exploit the available data.

## Method: BAC-PE

BAC-PE introduces a Q-function correction term derived from the behavior policy:

$$\hat{Q}^\pi(s, a) = Q^\pi(s, a) + \alpha \cdot A^\mu(s, a)$$

where $A^\mu(s, a) = Q^\mu(s, a) - V^\mu(s)$ is the **behavioral advantage**—the advantage of action $a$ under the behavior policy's value function. The intuition is:
- If $a$ is a good action under $\mu$ (positive behavioral advantage), the correction boosts $Q^\pi(s, a)$, mitigating conservatism.
- If $a$ is out-of-distribution and was rarely taken by $\mu$ (low $Q^\mu$), the correction suppresses $Q^\pi(s, a)$, mitigating overestimation.

The behavior policy's Q-function $Q^\mu$ is estimated from the offline data using standard temporal difference methods, as the behavior policy's distribution matches the data by construction.

## Diffusion Policy Representation

Both the behavior policy and the learned policy are represented as **diffusion models** over the action space. A diffusion policy parameterizes the action distribution as an iterative denoising process:

$$\pi_\theta(a | s) \propto \exp\!\left(-\frac{1}{2\sigma^2}\|a - \mu_\theta(s, t)\|^2\right)$$

where $\mu_\theta$ is the learned denoising network and $t$ is the diffusion timestep. Diffusion policies excel at representing multimodal action distributions (common in robotics and navigation tasks) that Gaussian policies or deterministic policies cannot capture.

Policy optimization maximizes the corrected Q-function while staying close to the behavior distribution through distribution matching:

$$\mathcal{L}(\theta) = -\mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\theta}\!\left[\hat{Q}^\pi(s, a)\right] + \lambda \cdot D_{\mathrm{KL}}(\pi_\theta(\cdot|s) \| \mu(\cdot|s))$$

## Theoretical Analysis

The paper provides convergence guarantees for BAC-PE:

**Theorem (informal):** Let $Q^\pi_k$ be the corrected Q-estimate at iteration $k$. Then for any $\epsilon > 0$, there exists $K$ such that for all $k > K$:
$$\left\|Q^\pi_k - Q^{\pi^*}\right\|_\infty \leq \epsilon + C \cdot \delta_{\mathrm{coverage}}$$

where $\delta_{\mathrm{coverage}}$ measures the support overlap between $\pi$ and $\mu$, and $C$ is a problem-dependent constant. Actions that are well-covered by the behavior distribution converge to their true Q-values; the error bound degrades gracefully as actions move out of distribution.

## Results

BAC-PE outperforms prior offline RL baselines including CQL, TD3+BC, and diffusion-based methods (DDPO, Diffusion-QL) on:
- **D4RL locomotion tasks** (HalfCheetah, Hopper, Walker2D, Ant): higher normalized scores on medium and medium-expert datasets
- **Antmaze navigation:** improved success rates over conservative baselines on the harder long-horizon tasks
- **Robosuite manipulation:** better task completion rates on pick-and-place variants

## Why It Matters

Offline RL is critical for learning from logged data in domains where online interaction is expensive or dangerous (robotics, healthcare, autonomous driving). BAC-PE offers a principled middle ground between over-conservative methods that fail to exploit good data and overoptimistic methods that exploit noise—and the diffusion policy representation handles the multimodal action distributions endemic to these domains.

## Key Contributions

1. Behavioral Advantage Corrected Policy Evaluation (BAC-PE) correcting Q-values with behavior-policy estimates
2. Theoretical convergence bound on Q-estimation error
3. Integration with diffusion policy for multimodal action distributions
4. Strong empirical results across D4RL, Antmaze, and manipulation benchmarks
