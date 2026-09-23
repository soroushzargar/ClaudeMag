# InsertAnything: Generalizable Contact-Rich Precision Insertion from Simulation to Reality

**arXiv:** 2609.24511  
**Date:** September 2026

---

## One-Line Summary

A single simulation-trained RL policy using only target poses and 3-axis fingertip force feedback achieves 95% success across eight unseen real-world insertion geometries at sub-millimeter clearances, with the first perfect score on ManipulationNet's peg-in-hole benchmark.

---

## Problem

Precision insertion—placing a peg or component into a hole with sub-millimeter clearance—is a fundamental assembly task in industrial robotics. It is contact-rich, which makes it sensitive to alignment errors, prone to jamming, and difficult to model analytically. The challenges compound across diverse geometries: a policy trained for a round peg often fails on a hexagonal or square one because the contact mechanics and jam-recovery strategies differ substantially.

The standard approach requires expensive real-world demonstrations per geometry, or domain randomization with explicit simulator-reality gap bridging. Neither scales to the combinatorial diversity of real assembly tasks.

---

## Why Existing Approaches Fall Short

- **Imitation learning from demonstrations** generalizes poorly across geometries; demonstrations must be re-collected for each new part
- **Model-based contact planning** requires accurate part-geometry models and contact-mode enumeration, which is intractable for arbitrary shapes
- **Vision-based RL** adds perception errors to the control errors and requires careful camera calibration
- **Tactile-rich sensors** (GelSight, TacTip) provide rich contact feedback but introduce high-dimensional observation spaces and complex sim-to-real transfer for the sensor model

InsertAnything uses minimal, easy-to-simulate force signals—3D fingertip force vectors—as the sole contact feedback, achieving generalization through policy design rather than sensing richness.

---

## Core Method

**Observation Space.** The policy observes:
1. A 6-DoF target pose for the end-effector (translation + quaternion)
2. Three-dimensional fingertip force readings from each finger tip (compact, low-dimensional)

No visual input is used. The target pose encodes the desired insertion direction; force feedback provides the search and jamming-recovery signal.

**Decoupled Gated Reward.** The reward function has two phases gated by an alignment threshold $\epsilon_{\text{align}}$:

$$r_t = \begin{cases}
r_{\text{align}}(s_t) & \text{if dist}(s_t, \text{hole}) > \epsilon_{\text{align}} \\
r_{\text{insert}}(s_t) & \text{otherwise}
\end{cases}$$

where $r_{\text{align}}$ rewards reducing lateral misalignment to the hole axis and $r_{\text{insert}}$ rewards downward insertion progress. The gate prevents the model from receiving premature insertion reward before the peg is aligned, avoiding a failure mode where it jams hard.

**Force-Signal Smoothing.** Raw fingertip force readings are averaged over a short sliding window to reduce noise without introducing significant lag:

$$\tilde{f}_t = \frac{1}{W}\sum_{w=0}^{W-1} f_{t-w}$$

**State-Independent Standard Deviations.** The policy outputs action means with a learned standard deviation that does not depend on the current state. This prevents entropy collapse in the stochastic policy during the low-clearance insertion phase, where small state changes could otherwise cause the policy to become near-deterministic and lose the exploratory behavior needed for jam recovery.

**Simulation Setup.** Trained entirely in MuJoCo on a hexagonal peg-in-hole task with randomized hole position (up to ±5 mm), randomized peg orientation (up to ±15°), and randomized friction. No real-world demonstrations are used; the sim-to-real gap for fingertip forces is handled by calibrated force scaling.

---

## Technical Formulation

Let $\mathbf{p} \in \mathbb{R}^7$ (target pose), $\mathbf{f} \in \mathbb{R}^{3n_f}$ (smoothed fingertip forces for $n_f$ fingers). Policy:

$$\pi_\theta: (\mathbf{p}, \tilde{\mathbf{f}}) \mapsto \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{p}, \tilde{\mathbf{f}}),\; \text{diag}(\boldsymbol{\sigma}^2))$$

where $\boldsymbol{\sigma}$ is state-independent.

Alignment reward with target $\mathbf{p}^* = (x^*, y^*, z^*)$:

$$r_{\text{align}} = -\sqrt{(x - x^*)^2 + (y - y^*)^2}$$

Insertion reward:

$$r_{\text{insert}} = z^* - z_t \quad \text{(downward progress toward target depth)}$$

Full episode return:

$$R = \sum_t \gamma^t r_t + R_{\text{success}} \cdot \mathbf{1}[\text{depth} \geq d_{\text{target}}]$$

---

## Learning Procedure

1. Randomize hole position, peg orientation, and friction in MuJoCo
2. Train PPO-based RL agent on hexagonal peg-in-hole only
3. Calibrate force scaling between simulator and real sensor
4. Deploy directly on real robot—no demonstrations, no fine-tuning

---

## What the Paper Claims

A policy trained on a single simulated geometry (hexagonal) achieves high success rates across geometries it has never seen (round, square, triangular, star-shaped, etc.) because force-guided search and jam recovery are geometry-agnostic skills. The decoupled gated reward is essential: ablating the gate reduces success by removing the alignment prerequisite.

---

## Experimental Findings

- **Main result:** 95.0% average success across 8 real-world insertion tasks with minimum clearance 0.02 mm
- **ManipulationNet benchmark:** First-ever perfect score of 20/20 under the Human-in-the-Loop evaluation protocol with fully autonomous motion
- **Cross-clearance generalization:** Policy trained on 0.1 mm clearance generalizes to 0.05 mm with modest degradation
- **Force reduction:** Peak contact force reduced by ~30% over the baseline without force feedback, reducing part wear

---

## Ablations and Interpretation

- **No force feedback (pose-only):** Success drops to ~40%; policy cannot recover from jams
- **No decoupled gate (pure sum reward):** Success drops to ~65%; early jams due to premature insertion attempts
- **State-dependent $\sigma$:** Success drops to ~70%; policy collapses to near-deterministic during insertion
- **No force smoothing:** Success drops to ~80% due to noisy force signals causing erratic recovery motions

---

## Reference

**InsertAnything: Generalizable Contact-Rich Precision Insertion from Simulation to Reality**. arXiv:2609.24511, September 2026.  
https://arxiv.org/abs/2609.24511
