# VTV-FM: Flow Matching through Variational Terminal-Velocity Closure

**arXiv:** 2610.00785  
**Venue:** NeurIPS 2026  
**Authors:** Haoyang Jiang, Yuheng Li, Di Yang, Yanhai Xiong, Haipeng Chen, Yi He  
**Date:** September–October 2026

---

## One-Line Summary

Static datasets lack the terminal-velocity information needed for second-order flow matching; VTV-FM recovers it via a minimum-acceleration energy principle, yielding a closed-form cubic bridge and improved sample quality over first-order and high-order baselines on CIFAR-10, physical fields, and low-dimensional benchmarks.

---

## Problem

Flow matching (FM) frames generative modeling as learning a velocity field $v_\theta(x, t)$ that transports noise samples $x_0 \sim p_0$ to data samples $x_1 \sim p_1$ along a continuous path $x_t$. The dominant paradigm uses **first-order (linear) bridges**:

$$x_t = (1 - t) x_0 + t x_1$$

with velocity $v(x_t, t) = x_1 - x_0$ constant along each path. This is computationally convenient but ignores **curvature**—the empirically observed property that real data distributions are not connected by straight lines.

Second-order flow matching would model both position and velocity, allowing curved paths that reduce the number of function evaluations (NFE) needed during sampling. But second-order bridges require **terminal velocities** $\dot{x}_1$ at the data endpoints, and static image datasets contain no such velocities.

---

## Key Observation: Minimum-Acceleration Energy

The authors observe that, while terminal velocities are unknown, the **minimum-acceleration** principle selects a unique velocity: the one that makes the path as smooth as possible (minimizes the integral of squared acceleration).

For a boundary-value ODE from $(x_0, \dot{x}_0)$ to $(x_1, \dot{x}_1)$ with fixed endpoints, the minimum-acceleration bridge is a **cubic Hermite spline**. The authors show that the terminal velocity $\dot{x}_1$ that minimizes the total squared acceleration along the cubic path has a closed-form solution in terms of $x_0$, $x_1$, and $\dot{x}_0$:

$$\dot{x}_1^* = 6(x_1 - x_0) - 3\dot{x}_0 \cdot \left(1 + \frac{1}{T}\right)$$

where $T$ is the total transport time and $\dot{x}_0$ is drawn from a Gaussian prior over initial velocities.

---

## Why Existing Approaches Fall Short

- **Linear FM (Lipman et al. 2022, Liu et al. 2022):** Uses straight-line paths; requires many NFE to follow curved data manifolds during sampling.
- **Stochastic interpolants / DDPM:** Introduce noise along the path; good for sample diversity but not for deterministic low-NFE sampling.
- **High-order FM (Ho et al. 2023):** Proposes higher-order paths but requires paired velocity measurements $(\dot{x}_0, \dot{x}_1)$ from the dataset; unavailable for static image datasets.
- **Optimal-transport FM (Tong et al. 2023):** Finds globally optimal pairings but uses linear bridges, ignoring curvature.
- **VTV-FM:** The first method that derives second-order bridges from minimum energy without requiring velocity annotations.

---

## Core Mechanism

VTV-FM replaces the linear bridge with a cubic Hermite spline parameterized by:
- Source point $x_0$ and its initial velocity $\dot{x}_0 \sim \mathcal{N}(0, \sigma_v^2 I)$
- Target point $x_1$ and its derived terminal velocity $\dot{x}_1^*$ (from the closed-form minimum-acceleration solution)

The cubic bridge $\gamma_t$ is:

$$\gamma_t = h_{00}(t) x_0 + h_{10}(t) \dot{x}_0 + h_{01}(t) x_1 + h_{11}(t) \dot{x}_1^*$$

where $h_{00}, h_{10}, h_{01}, h_{11}$ are the Hermite basis polynomials evaluated at time $t \in [0, 1]$.

The velocity field to be learned is:

$$v_\theta(x, t) \approx \dot{\gamma}_t = h'_{00}(t) x_0 + h'_{10}(t) \dot{x}_0 + h'_{01}(t) x_1 + h'_{11}(t) \dot{x}_1^*$$

and a second network predicts the acceleration $a_\theta(x, t) \approx \ddot{\gamma}_t$ to enable second-order ODE solvers.

---

## Technical Formulation

The minimum-acceleration energy functional:

$$E[\gamma] = \int_0^T \|\ddot{\gamma}_t\|^2 \, dt$$

Minimizing over cubic Hermite paths subject to $\gamma_0 = x_0, \gamma_T = x_1, \dot{\gamma}_0 = \dot{x}_0$ yields:

$$\dot{x}_1^* = \arg\min_{\dot{x}_1} E[\gamma] = \frac{6(x_1 - x_0)}{T} - 3\dot{x}_0$$

The training objective for the velocity network:

$$\mathcal{L}_v(\theta) = \mathbb{E}_{t, x_0, x_1, \dot{x}_0}\!\left[\|v_\theta(\gamma_t, t) - \dot{\gamma}_t\|^2\right]$$

The optimal transport pairing cost is also extended:

$$C_{\text{VTV}}(x_0, x_1) = E[\gamma^*(x_0, x_1)] = \frac{2\|x_1 - x_0\|^2}{T^2} - \frac{2 \langle \dot{x}_0, x_1 - x_0\rangle}{T} + \|\dot{x}_0\|^2 \cdot \frac{T}{3}$$

which reduces to the standard L2 cost when $\dot{x}_0 = 0$.

---

## Learning or Inference Procedure

1. Sample source $x_0 \sim p_0$ (Gaussian noise), target $x_1 \sim p_1$ (data), and initial velocity $\dot{x}_0 \sim \mathcal{N}(0, \sigma_v^2 I)$.
2. Compute terminal velocity $\dot{x}_1^*$ using the closed-form formula.
3. Evaluate the cubic bridge $\gamma_t$ and its derivatives at a random time $t \in [0, 1]$.
4. Train $v_\theta$ and $a_\theta$ with the regression objectives $\mathcal{L}_v$ and $\mathcal{L}_a$.
5. At inference, integrate the learned second-order ODE from $x_0$ to $x_1$ using a velocity-Verlet or Runge-Kutta-4 solver with as few as 5 NFE.

---

## What the Paper Claims

- VTV-FM produces smoother transport paths (lower mean curvature) than linear FM, enabling fewer NFE at matched sample quality.
- The closed-form terminal velocity is sufficient to match or exceed learned velocity estimates on all evaluated datasets.
- The extended OT cost provides better coupling on anisotropic data distributions.

---

## Experimental Findings

- **Low-dimensional:** On 2D toy distributions (checkerboard, moons, spirals), VTV-FM reduces mean curvature by 42% and improves Wasserstein-2 at 5 NFE by 31% vs. linear FM.
- **PDE-governed fields:** On Navier-Stokes flow prediction, VTV-FM achieves lower RMSE than DDPM and linear FM at all NFE budgets; at 5 NFE, RMSE is 0.042 vs. 0.067 for linear FM.
- **CIFAR-10:** VTV-FM achieves FID 2.83 at 50 NFE with a standard U-Net architecture, vs. 3.11 for linear FM and 3.04 for rectified flow. At 5 NFE, FID is 7.21 vs. 9.48 for linear FM.
- **Training cost:** VTV-FM adds approximately 15% training time due to the acceleration prediction head; inference is the same as linear FM if only $v_\theta$ is used.

---

## Ablations and Interpretation

- **$\dot{x}_0 = 0$ (zero initial velocity):** Cubic paths degenerate to symmetric Hermite splines; performance slightly below the full VTV-FM with sampled $\dot{x}_0$.
- **$\sigma_v$ sensitivity:** FID on CIFAR-10 is stable for $\sigma_v \in [0.5, 2.0]$; outside this range, paths become too curved (large $\sigma_v$) or too straight (small $\sigma_v$).
- **Acceleration prediction head:** Using the second-order solver with $a_\theta$ reduces NFE by 40% at FID 3.0 compared to using only $v_\theta$ with a first-order solver.
- **OT coupling:** VTV-FM with the extended OT cost reduces FID by 0.22 on CIFAR-10 vs. using the standard L2 cost.
- **Curvature analysis:** Path curvature correlates inversely with the TV distance between $p_0$ and $p_1$; the minimum-acceleration principle automatically assigns lower curvature to well-matched pairs.

---

## Reference

Haoyang Jiang, Yuheng Li, Di Yang, Yanhai Xiong, Haipeng Chen, Yi He. **VTV-FM: Flow Matching through Variational Terminal-Velocity Closure**. arXiv:2610.00785. NeurIPS 2026.  
https://arxiv.org/abs/2610.00785
