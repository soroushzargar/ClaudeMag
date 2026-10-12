# Differentiable Systematic Resampling for Variational Sequential Monte Carlo

**arXiv:** 2610.12094  
**Authors:** Fredrik Cumlin, Saikat Chatterjee  
**Venue:** NeurIPS 2026  
**Date:** October 8, 2026

---

## One-Line Summary

Differentiable Systematic Resampling (DSR) makes the variance-optimal systematic resampling step of particle filters differentiable by softening its CDF-level ancestor assignments with sigmoid functions and sorting particles via Morton curves, with bias decaying exponentially in the temperature parameter.

---

## Problem

Sequential Monte Carlo (SMC) methods approximate posterior distributions in latent variable models by maintaining a set of weighted particles $\{(w_t^k, x_t^k)\}_{k=1}^K$. **Resampling** is the step that selects ancestors for the next generation of particles, preventing particle degeneracy. Systematic resampling is the most variance-efficient scheme, but its use of a step-function CDF produces discrete, non-differentiable ancestor assignments. This blocks gradient-based optimisation of model parameters in **variational SMC** (VSMC) frameworks, where maximising the evidence lower bound (ELBO) requires gradients through the full particle trajectory.

---

## Why Existing Approaches Fall Short

- **Multinomial resampling** is differentiable via the straight-through estimator, but has higher variance than systematic resampling.
- **Optimal-transport (OT) based differentiable resampling** (e.g., using the Sinkhorn algorithm) preserves the efficiency of systematic resampling, but requires iterative solvers at every resampling step, incurring substantial computational overhead.
- **Stop-gradient / detached resampling** avoids differentiating through resampling altogether, biasing the gradient estimator.
- **Gumbel-top-$k$ tricks** apply to multinomial sampling but do not directly generalise to the CDF-ordered, banded structure of systematic resampling.

---

## Core Method

Differentiable Systematic Resampling (DSR) softens systematic resampling at the CDF level:

1. **CDF softening:** The ancestor assignment of particle $i$ to offspring $j$ in standard systematic resampling is determined by a step function of the cumulative weight sum. DSR replaces the step function with a sigmoid:
   $$A_{ij}(\tau) = \sigma\!\left(\frac{W_{\leq i} - U_j}{\tau}\right)$$
   where $W_{\leq i}$ is the prefix weight sum up to particle $i$, $U_j$ is the $j$-th evenly spaced quantile, and $\tau > 0$ is a temperature.

2. **Morton curve sorting:** Particles are reordered along a Morton (Z-order) space-filling curve before resampling, so that spatially proximate particles are adjacent in the sorted order. This preserves the "banded" ancestor matrix property of systematic resampling (each offspring is assigned to one of a small number of nearby ancestors), which is key to its low variance.

3. **Temperature annealing:** As $\tau \to 0$, the soft assignment converges to the hard systematic resampling; the bias vanishes **exponentially** in $1/\tau$.

---

## Technical Formulation

Let $\{w^k\}_{k=1}^K$ be normalised weights with prefix sums $W_{\leq k} = \sum_{j \leq k} w^j$. Systematic resampling draws a single $U \sim \text{Uniform}(0, 1/K)$ and assigns ancestor $a^j = \min\{i : W_{\leq i} \geq (j-1)/K + U\}$.

DSR defines the soft ancestor matrix:
$$\tilde{A}_{ij} = \sigma\!\left(\frac{W_{\leq i} - \frac{j-1}{K} - U}{\tau}\right) - \sigma\!\left(\frac{W_{\leq i-1} - \frac{j-1}{K} - U}{\tau}\right)$$

This gives a differentiable (soft) assignment $\tilde{x}^j = \sum_i \tilde{A}_{ij}\, x^i$.

**Bias bound:** $\|\mathbb{E}[\tilde{A}] - \mathbb{E}[A]\|_\infty \leq C \exp(-c/\tau)$ for constants $C, c > 0$.

---

## Learning or Inference Procedure

**In VSMC:**
1. At each time step $t$, propagate particles through the proposal: $x_t^k \sim q_\phi(x_t \mid x_{t-1}^k, y_t)$.
2. Compute and normalise importance weights $w_t^k \propto p_\theta(y_t \mid x_t^k) p_\theta(x_t^k \mid x_{t-1}^k) / q_\phi(x_t^k \mid x_{t-1}^k, y_t)$.
3. Apply DSR to produce the soft-resampled particle set $\{\tilde{x}_t^k\}$.
4. Accumulate the ELBO: $\mathcal{L} = \mathbb{E}\!\left[\log \frac{1}{K}\sum_k w_T^k\right]$.
5. Backpropagate through the soft resampling steps to update $\phi, \theta$.

---

## What the Guarantee Says

DSR provides a **theoretical bias bound**: the soft ancestor assignment converges to the hard systematic resampling assignment at an exponential rate in $1/\tau$. This is stronger than the polynomial convergence of OT-based methods at equivalent temperature. The proof uses the fact that the sigmoid approximates the Heaviside step function with exponentially small pointwise error.

---

## Experimental Findings

- On **stochastic Lorenz-96 dynamics** and related systems, DSR matches OT-based differentiable resampling on filtering NLL while using significantly less wall-clock time per step.
- On **real handwriting generation** (IAM dataset), DSR improves log-likelihood over both multinomial resampling and OT-based approaches.
- DSR is **3–5× faster** per resampling step than OT-based alternatives due to the absence of iterative solvers.
- The Morton-curve sorting is shown to be necessary: naïve soft resampling without it produces a dense ancestor matrix with higher variance.

---

## Ablations and Interpretation

**Without Morton sorting:** The soft ancestor matrix loses its banded structure, gradient variance increases, and ELBO estimates degrade.

**Temperature sensitivity:** Below a threshold $\tau^*$ the gradients become noisy (near-discrete); above it bias becomes appreciable. The optimal $\tau$ is problem-dependent and can be found via a short sweep.

**Number of particles $K$:** DSR's advantage over multinomial resampling grows with $K$, consistent with the systematic resampling advantage being $O(1/K^2)$ in particle variance vs. $O(1/K)$.

**Comparison with stop-gradient:** Stop-gradient resampling shows a systematic bias in the ELBO that is eliminated by DSR.

---

## Reference

Fredrik Cumlin, Saikat Chatterjee.
**Differentiable Systematic Resampling for Variational Sequential Monte Carlo.**
NeurIPS 2026. arXiv:2610.12094, October 2026.
https://arxiv.org/abs/2610.12094
