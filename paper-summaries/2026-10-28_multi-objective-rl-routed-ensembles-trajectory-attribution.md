# Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution

**Authors:** Zhenshuo Zhang, Ruoxuan Xiong, Hongyang R. Zhang  
**Venue:** NeurIPS 2026  
**arXiv:** 2610.35733  
**Date:** October 26, 2026  
**URL:** https://arxiv.org/abs/2610.35733

---

## Summary

Multi-objective reinforcement learning (MORL) trains agents to satisfy multiple reward objectives simultaneously, balancing them according to a preference weight vector. The challenge is that objectives often conflict: improving performance under one preference setting can harm others. This paper provides both a theoretical analysis of objective conflict and practical algorithms that exploit conflict structure to learn a diverse set of preference-conditioned policies.

---

## Problem Setting

In MORL, the agent receives a vectorized reward **r** = (r₁, ..., rₖ) and a preference weight **w** ∈ Δᵏ⁻¹. The goal is to find a policy π that maximizes the expected scalarized return **w**ᵀ J(π) simultaneously across many preference weights. Naive approaches train a single conditioned policy, which fails when objectives conflict strongly.

---

## Contributions

### 1. Objective-Conflict Coefficients

The authors define objective-conflict coefficients that quantify how much pairs of reward objectives pull policy gradients in opposing directions. These coefficients are computable from rollout data and predict which preference weight regions are hardest to cover with a single policy. Theoretical regret bounds for online MORL are derived as a function of the maximum conflict coefficient.

### 2. Routed Ensemble Policy Learning

Rather than training a single policy conditioned on **w**, the method maintains a small ensemble of K basis policies {π₁, ..., π_K}. At each iteration:

1. Each collected trajectory τ is routed to the basis policy πₖ with the highest expected return under τ's preference weight.
2. Each πₖ is updated with PPO using only the trajectories routed to it.

Routing creates a natural specialization: each basis policy covers a neighborhood of the preference space where it is strongest, automatically adapting to conflict structure without manual partitioning.

### 3. Trajectory Attribution

A kernel ridge regression surrogate estimates policy performance as a function of which trajectories are included in the training set. This allows the method to compute the marginal value of each trajectory for each preference objective—high-value trajectories for underserved preference regions are upweighted in the training distribution.

---

## Key Results

- Consistent improvement over single-policy and standard ensemble baselines on multi-objective MuJoCo locomotion benchmarks
- Regret bounds that scale with the maximum objective-conflict coefficient, providing the first formal characterization of MORL hardness in terms of objective geometry
- Ablations confirming that both routing and trajectory attribution contribute independently

---

## Significance

The work provides the first principled connection between the geometry of objective conflict and the design of ensemble policies. The routed ensemble approach is simple to implement on top of any single-objective RL algorithm and scales gracefully as the number of preference weights grows.

---

## Citation

Zhenshuo Zhang, Ruoxuan Xiong, Hongyang R. Zhang.  
**Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution.**  
arXiv:2610.35733. NeurIPS 2026.  
https://arxiv.org/abs/2610.35733
