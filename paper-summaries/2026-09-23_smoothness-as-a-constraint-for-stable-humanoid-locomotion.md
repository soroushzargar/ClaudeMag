# Smoothness as a Constraint for Stable Humanoid Locomotion

**arXiv:** 2609.24552  
**Authors:** Utsav Panchal, Denis Kleyko, Unal Artan, Amy Loutfi  
**Date:** September 2026

---

## One-Line Summary

DeCap decouples whole-body smoothness into upper- and lower-body constraint groups with different tightness levels, reducing real-robot upper-body action rate by 2.50× and acceleration by 2.18× over reward-based baselines while transferring to new terrains without reward retuning.

---

## Problem

Whole-body humanoid locomotion requires balancing task performance (walking speed, stability, terrain traversal) with motion quality (smooth, human-like actuation that reduces mechanical wear and improves balance). The dominant approach is to add smoothness penalty terms to the RL reward function. This approach has two fundamental problems:

1. **Objective competition:** Smoothness penalties compete with task rewards during optimization; increasing one degrades the other, requiring delicate reward tuning for each deployment context.
2. **Uniform treatment:** A single smoothness term is applied to the whole body, ignoring the fundamentally different roles of the upper and lower body. The lower body must remain reactive to ground contact perturbations; the upper body must be tightly controlled to preserve the center-of-mass trajectory and prevent tipping.

---

## Why Existing Approaches Fall Short

- **Reward-based smoothness** (action-rate penalties, torque-change penalties) treats smoothness as a soft objective, making it prone to being traded away for task performance during RLVR
- **Lipschitz-constrained policies** (prior work) constrain the policy's sensitivity globally, without decoupling upper and lower body
- **Deterministic reference-tracking** controllers (MPC, WBC) provide smooth motion but require accurate dynamics models and cannot adapt to unknown terrain perturbations

---

## Core Method: DeCap

**Decoupled Constraint-aware Policy (DeCap)** reformulates smoothness as a constrained optimization problem, with separate constraint groups for the upper and lower body.

**Constraint Groups.** Let $\mathbf{a}_t^U \in \mathbb{R}^{n_U}$ and $\mathbf{a}_t^L \in \mathbb{R}^{n_L}$ denote the upper- and lower-body joint actions at time $t$. DeCap defines smoothness in terms of action rate:

$$\dot{\mathbf{a}}_t^U = \frac{\mathbf{a}_t^U - \mathbf{a}_{t-1}^U}{\Delta t}, \quad \dot{\mathbf{a}}_t^L = \frac{\mathbf{a}_t^L - \mathbf{a}_{t-1}^L}{\Delta t}$$

Constraints:

$$\|\dot{\mathbf{a}}_t^U\|_2 \leq \kappa_U, \quad \|\dot{\mathbf{a}}_t^L\|_2 \leq \kappa_L$$

with $\kappa_U \ll \kappa_L$: the upper body is tightly constrained; the lower body is given more room to respond to ground contacts.

**Bounded Barrier Penalty.** To handle these constraints within an RL framework (rather than constrained optimization), DeCap uses a bounded barrier function that activates *proactively* as the constraint limit is approached:

$$\phi(c, \kappa) = \begin{cases}
0 & c \leq \alpha \kappa \\
\frac{(c - \alpha\kappa)^2}{\kappa - \alpha\kappa} & \alpha\kappa < c < \kappa \\
\kappa - \alpha\kappa & c \geq \kappa
\end{cases}$$

where $c = \|\dot{\mathbf{a}}_t\|_2$, $\kappa$ is the constraint limit, and $\alpha \in (0,1)$ is the activation threshold fraction. The barrier activates in the range $[\alpha\kappa, \kappa)$ and is *bounded* (saturates at the limit value) to avoid catastrophic reward signal at violations.

The RL objective is:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{task}}(\theta) - \lambda_U \phi_U(\theta) - \lambda_L \phi_L(\theta)$$

---

## Technical Formulation

Upper-body barrier term added to reward:

$$R_{\text{smooth}}^U = -\lambda_U \cdot \phi(\|\dot{\mathbf{a}}_t^U\|_2,\; \kappa_U)$$

Lower-body barrier term:

$$R_{\text{smooth}}^L = -\lambda_L \cdot \phi(\|\dot{\mathbf{a}}_t^L\|_2,\; \kappa_L)$$

Joint action-rate constraint is also applied to acceleration (second differences):

$$\ddot{\mathbf{a}}_t^U = \frac{\dot{\mathbf{a}}_t^U - \dot{\mathbf{a}}_{t-1}^U}{\Delta t}$$

with an analogous barrier $\phi_{\text{acc}}(\|\ddot{\mathbf{a}}_t^U\|_2, \kappa_U^{\text{acc}})$.

---

## Learning Procedure

1. Define upper/lower body joint sets based on robot URDF
2. Set tight constraint $\kappa_U$ (derived from human motion capture statistics) and loose $\kappa_L$ (set to allow reactive stepping)
3. Train with PPO in Isaac Gym / Genesis on flat ground with domain randomization
4. Deploy the trained policy directly on the real humanoid; evaluate on flat ground and novel terrains
5. Test terrain transfer without any reward retuning

---

## What the Paper Claims

Reformulating smoothness as explicit constraints (rather than reward terms) provides direct control over the physical quantities responsible for smooth behavior. Decoupling removes the false assumption that upper and lower body have the same smoothness requirements. The bounded barrier avoids the numerical instabilities of standard log-barrier methods in RL.

---

## Experimental Findings

- **Upper-body action rate:** 2.50× reduction over reward-based smoothness baseline on flat ground
- **Upper-body acceleration:** 2.18× reduction over reward-based baseline
- **Lower-body smoothness:** Also improves marginally; the reactive range is preserved without degradation
- **Transient motion:** Reduced oscillations in torso and arm joints during gait transitions
- **Terrain transfer:** Constraints trained on flat ground transfer to stairs, slopes, and stepping stones without reward retuning; reward-based baselines require significant re-tuning

---

## Ablations and Interpretation

- **No decoupling (single constraint group):** Upper-body smoothness improves but lower-body reactivity degrades on uneven terrain, leading to falls
- **Standard log-barrier:** Training instability; gradient explosions near constraint limits
- **Reward term instead of constraint:** Smoothness improvement is ~40% of DeCap's improvement at matched task reward coefficient
- **$\alpha$ sensitivity:** $\alpha = 0.7$ provides the best balance; smaller $\alpha$ (earlier activation) slightly over-constrains and reduces task performance

---

## Reference

Utsav Panchal, Denis Kleyko, Unal Artan, Amy Loutfi. **Smoothness as a Constraint for Stable Humanoid Locomotion**. arXiv:2609.24552, September 2026.  
https://arxiv.org/abs/2609.24552
