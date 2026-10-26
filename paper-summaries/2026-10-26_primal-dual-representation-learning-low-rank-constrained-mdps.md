# Primal-Dual Representation Learning for Low-Rank Constrained MDPs

**Venue:** NeurIPS 2026 (Poster #151600)  
**Authors:** Chenhao Zhou, Chao Zhang, Wensong Bai, Hanbin Zhao, Hui Qian  
**Date:** 2026

---

## One-Line Summary

The paper presents the first sample-efficient, theoretically guaranteed representation-learning algorithms for low-rank CMDPs with continuous state spaces—separately handling soft (cumulative) and hard (per-episode) violation constraints via two primal-dual schemes.

---

## Problem

Safe reinforcement learning (RL) requires an agent to maximize return while respecting constraints on cumulative or per-episode costs. Practical safe RL environments have continuous, high-dimensional state spaces (e.g., robotic systems, autonomous vehicles), where tabular methods fail. Low-rank MDP structure—where the transition kernel factorizes as a product of low-dimensional feature representations—is a natural and theoretically tractable model for such settings. However, combining representation learning with constraint satisfaction is significantly harder than either alone: the representation and the dual variable for the constraint must be learned simultaneously, and errors in one can invalidate the guarantees of the other. No prior work provided end-to-end guarantees for this combined setting.

---

## Key Observation: Decoupling Representation Learning from Dual Updates Is Sufficient

In an unconstrained low-rank MDP, representation learning by maximum likelihood from an exploratory policy achieves near-optimal feature error. In a constrained setting, the Lagrangian dual variable penalizes constraint violations but does not change the structure of the transition kernel. Therefore, one can learn the representation independently, using the same exploration strategy as in the unconstrained case, and plug the learned features into a primal-dual planning step that handles the constraint. This decoupling is the key insight enabling clean, composable theoretical bounds.

---

## Why Existing Approaches Fall Short

- **Tabular constrained RL:** Provides finite-sample guarantees but is computationally intractable for continuous state spaces.
- **Linear function approximation for CMDPs:** Requires the true features to be given; cannot handle the setting where features must be learned from data.
- **Deep safe RL (CPO, IPO, PDO):** Achieves empirical success but lacks finite-sample or regret guarantees.
- **Unconstrained low-rank MDP representation learning:** Solves the feature learning problem but cannot enforce safety constraints; the techniques do not extend directly because the dual variable interacts with the value function estimates in non-trivial ways.

---

## Core Methods: REP-PD-TEG and REP-PD-H-TEG

The paper proposes two algorithms:

**REP-PD-TEG (soft/cumulative violation):**
- Uses a decaying epsilon-greedy exploration policy to collect data for maximum-likelihood feature learning.
- Constructs optimistic estimates of both the return and the constraint utility using UCB-style bonus functions.
- Plans with an optimistic Lagrangian objective: $\max_\pi \min_\lambda \hat{V}^\pi - \lambda (\hat{C}^\pi - b)$, where $b$ is the budget and $\lambda \geq 0$ is the dual variable.
- Updates the dual variable by gradient ascent on the (negative) optimistic Lagrangian.

**REP-PD-H-TEG (hard/certified per-episode violation):**
- Adds a lower-confidence constraint certificate: at each episode, it verifies that the optimistic policy's constraint value under the pessimistic feature estimate remains below the budget.
- Uses a stabilized dual search that shrinks the dual step size when the certificate is tight, preventing oscillation around the constraint boundary.
- Guarantees that the deployed policy never violates the constraint with high probability, on a per-episode basis.

---

## Technical Formulation

The low-rank CMDP structure assumes transition probability $P(s' \mid s, a) = \phi(s, a)^\top \mu(s')$, where $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$ are state-action features and $\mu: \mathcal{S} \to \mathbb{R}^d$ is a measure over next states, with $d \ll |\mathcal{S}|$.

The constrained MDP problem is:

$$\max_\pi V^r_\pi \quad \text{s.t.} \quad V^c_\pi \leq b$$

where $V^r_\pi$ and $V^c_\pi$ are the return and constraint utility under policy $\pi$.

For REP-PD-TEG, the estimated features $\hat{\phi}$ from maximum-likelihood minimization satisfy $\|\hat{\phi} - \phi^*\|_\infty \leq \epsilon_\phi$ with high probability after $\tilde{O}(d^2 / \epsilon_\phi^2)$ samples. The optimistic Lagrangian planning step then achieves a return regret of:

$$\text{Reg}_K = \tilde{O}\!\left(\sqrt{K}\right) \quad \text{and total samples} \quad \tilde{O}\!\left(K^{3/4}\right)$$

where $K$ is the number of episodes and the $\tilde{O}$ suppresses logarithmic factors.

For REP-PD-H-TEG, the per-episode hard constraint is:

$$V^c_{\pi_k} \leq b \quad \text{for all } k = 1, \ldots, K$$

with high probability, at the cost of a slightly worse return bound due to the conservative constraint certificate.

---

## Learning or Inference Procedure

**REP-PD-TEG:**
1. Run decaying-epsilon-greedy exploration for $T_0$ episodes to collect feature-learning data.
2. Fit $\hat{\phi}$ by maximum likelihood on the collected transitions.
3. For each subsequent episode: compute optimistic value and constraint estimates using $\hat{\phi}$ and UCB bonuses; solve the optimistic Lagrangian; deploy the resulting policy; update $\lambda$ by dual gradient ascent.

**REP-PD-H-TEG:**
1. Steps 1–2 as above.
2. Before deploying a policy, certify its constraint value using a pessimistic lower bound $\underline{V}^c_\pi$.
3. If the certificate passes ($\underline{V}^c_\pi \leq b$), deploy; otherwise invoke the stabilized dual search to find a feasible policy.
4. Repeat with the same feature estimate; re-fit features periodically to reduce feature error.

---

## What the Paper Claims

- **First theoretical guarantees:** REP-PD-TEG and REP-PD-H-TEG are claimed to be the first algorithms with provable finite-sample guarantees for representation learning in low-rank CMDPs with continuous state spaces.
- **Soft violation:** Õ(√K) return regret and Õ(K^{3/4}) total sample complexity for REP-PD-TEG.
- **Hard violation:** Per-episode constraint satisfaction with high probability for REP-PD-H-TEG.
- **Decoupling enables composability:** The representation-learning and primal-dual components are independently analyzed, making the results modular and potentially applicable to other constrained settings.

---

## Experimental Findings

The paper provides both theoretical analysis and experiments on continuous-state safe RL environments:

- **Environments:** Continuous-state safety gridworld (continuous analog of SafetyGym), constrained LQR (linear quadratic regulator with safety constraints).
- **Baselines:** Tabular constrained RL (discretized), CPO, and PDO with fixed features.
- **Key result:** REP-PD-TEG achieves competitive return while keeping cumulative constraint violations within budget; REP-PD-H-TEG achieves zero per-episode violations after feature learning converges, at the cost of slightly lower return.
- **Benefit of learned features:** Methods with fixed (incorrect) features fail to satisfy constraints; REP-PD-TEG's learned features enable robust constraint satisfaction.

---

## Ablations and Interpretation

- **Fixed vs. learned features:** Using the ground-truth features in the primal-dual step achieves near-optimal performance; using fixed incorrect features degrades both return and constraint satisfaction, confirming the value of end-to-end representation learning.
- **Epsilon-greedy exploration rate:** The decaying schedule is important; a constant rate leads to over-exploration and slower convergence; an aggressive decay leads to under-exploration and poor feature quality.
- **Dual step size stability (REP-PD-H-TEG):** The stabilized dual search is critical for hard-violation safety; naive gradient ascent on the dual variable oscillates around the constraint boundary and causes intermittent violations.

---

## Reference

Chenhao Zhou, Chao Zhang, Wensong Bai, Hanbin Zhao, Hui Qian. **Primal-Dual Representation Learning for Low-Rank Constrained MDPs**. NeurIPS 2026.  
https://neurips.cc/virtual/2026/poster/151600
