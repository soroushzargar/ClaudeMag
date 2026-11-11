# Reward-Estimated Hypergradient for Bilevel Reinforcement Learning with Black-Box Follower

**arXiv:** 2611.11998  
**Authors:** NeurIPS 2026 Submission (anonymous)  
**Venue:** NeurIPS 2026  
**Submitted:** 9 November 2026  
**URL:** https://arxiv.org/abs/2611.11998

---

## Summary

Bilevel reinforcement learning (bilevel RL) models a Stackelberg game between a leader and a follower: the leader optimizes a meta-policy whose parameters influence the follower's best-response policy. Computing the leader's hypergradient — the gradient of the leader's objective with respect to the leader's parameters, taking into account how those parameters change the follower's best response — requires access to the follower's reward function. Existing bilevel RL methods assume white-box access to this reward. RE-HG (Reward-Estimated Hypergradient) removes this assumption by using inverse RL to estimate the follower's reward from observed trajectories, then computing the hypergradient from the estimate. It further addresses two failure modes of naive IRL-based estimation: reward-shaping bias and Hessian ill-conditioning.

---

## Method

**Bilevel RL formulation.** Let θ_L be the leader's parameters and θ_F*(θ_L) be the follower's best-response policy. The leader's objective is V_L(θ_L, θ_F*(θ_L)), and the hypergradient is:
  ∇_θ_L V_L = ∂V_L/∂θ_L + (∂θ_F*/∂θ_L)^T ∇_{θ_F} V_L

Computing ∂θ_F*/∂θ_L requires differentiating through the follower's best-response optimality condition, which in turn requires R_F — the follower's reward — which is unknown in the black-box setting.

**Step 1: Reward estimation via IRL.** RE-HG runs maximum-entropy IRL on observed follower trajectories to recover an estimated reward R̂_F. The estimated reward is then used as a proxy for R_F in the hypergradient computation.

**Step 2: Bias cancellation.** A fundamental problem with IRL-estimated rewards is that they are only identified up to potential-based reward shaping: any R̂_F + Φ(s') - Φ(s) fits the observed trajectories equally well. This ambiguity introduces a systematic bias in the hypergradient. RE-HG's bias-cancellation mechanism aggregates gradient information across trajectories from multiple, diverse environmental dynamics (different initial conditions, transition perturbations). The shaping functions cancel in expectation across sufficiently diverse dynamics, while the true reward signal accumulates.

**Step 3: Eigenvalue truncation.** Computing the hypergradient involves inverting the Hessian of the follower's value function at the estimated reward. Near reward landscape extrema, the Hessian can be near-singular, causing hypergradient norm explosions. RE-HG applies eigenvalue truncation: Hessian eigenvalues below a threshold ε are set to ε before inversion, acting as implicit regularization.

---

## Experimental Setup

RE-HG is evaluated on bilevel RL benchmarks where:
- The follower has a known reward (white-box oracle) for ground-truth comparison
- RE-HG and baselines observe only follower trajectories (black-box)
- Tasks include navigation, manipulation, and mechanism design environments with diverse reward landscapes

---

## Results

| Setting | RE-HG | White-Box Oracle | Single-Level Baseline |
|---------|-------|-----------------|----------------------|
| Smooth reward landscape | ≈ Oracle | — | -15% |
| Sharp reward landscape | Oracle + 8% | — (stuck in local min) | -32% |
| Gradient alignment (cosine) | 0.91 | 1.0 | 0.63 |

RE-HG produces hypergradient estimates with 0.91 average cosine similarity to the oracle (white-box) gradient. On sharp reward landscapes — where the oracle's exact Hessian inversion gets trapped in poor local minima — RE-HG's eigenvalue truncation regularizes the inversion and achieves better final performance than the oracle.

---

## Key Insight

The key observation is that reward shaping, while a nuisance in single-level IRL, becomes a systematic bias source when rewards are used inside a hypergradient computation. By diversifying the dynamics under which trajectories are collected and aggregating gradients across this diversity, RE-HG exploits the mathematical structure of potential-based shaping (which depends only on states, not on trajectory paths) to cancel the bias while preserving the true reward signal — a technique with no analogue in standard IRL or standard bilevel optimization.

---

## Limitations

- Collecting trajectories across diverse environmental dynamics requires a controllable environment; real-world applications with a fixed, unmodifiable environment cannot exploit the full bias-cancellation mechanism.
- The IRL estimation step requires sufficient trajectory diversity; a follower playing in a low-entropy regime may not reveal enough information for good reward estimation.
- The eigenvalue truncation threshold ε is a hyperparameter; its optimal value is problem-dependent and must be tuned.
