# Error-Corrected Inference-Time Scaling for Imperfect Diffusion Models

**arXiv:** 2610.01933  
**Authors:** Zuokai Wen, Louis Grenioux, Weinan E, Jiequn Han  
**Affiliation:** Princeton University; Flatiron Institute, Simons Foundation  
**Date:** October 1, 2026

---

## One-Line Summary

By deriving Feynman-Kac dynamics that correct diffusion model errors on the fly, the Energy-based Feynman-Kac Corrector (EBFKC) achieves accurate molecular free-energy profiles and target distribution matching where standard inference-time scaling with more particles fails.

---

## Problem

Inference-time scaling for diffusion models works by running more particles through a sequential Monte Carlo (SMC) procedure to better approximate a target distribution or to steer generation toward a desired reward signal. However, this approach rests on an assumption that the pretrained diffusion model is exact—that its learned score function precisely captures the data distribution. In practice, data sparsity, modeling limitations, and finite training budgets make the model imperfect: its score function carries systematic error that no amount of additional Monte Carlo particles can remove. The result is that standard inference-time scaling inherits the model's bias even as Monte Carlo variance is reduced.

---

## Why Existing Approaches Fall Short

- **Monte Carlo only reduces variance, not bias:** Adding more particles in standard SMC reduces the statistical error of the particle approximation, but the inherent mismatch between the model's learned score and the true data score remains as a fixed systematic error regardless of particle count.
- **Path tracking error:** Existing methods may fail to track the prescribed probability path (e.g., a data-to-noise interpolant) when the model is imperfect, accumulating trajectory-level errors that compound over the diffusion timesteps.
- **Reward tilting without correction:** Reward-guided diffusion (e.g., for molecule generation) typically reweights samples under the assumption of a correct model; an imperfect model introduces additional bias into the tilted distribution that existing guidance methods cannot remove.

---

## Core Method

**Energy-based Feynman-Kac Corrector (EBFKC).** The central insight is that the Feynman-Kac formula from stochastic analysis provides a mathematically principled mechanism to adjust the dynamics of a diffusion process so that it tracks a prescribed probability path exactly, even when the generator of the nominal dynamics (i.e., the pretrained diffusion model) is incorrect.

Concretely, EBFKC:
1. Specifies a target path as an energy-based model $\pi_t \propto \exp(-E_t(\mathbf{x}))$ parameterized by a reference energy $E_t$.
2. Derives a Feynman-Kac potential $G_t$ that, when applied as a multiplicative correction to the nominal diffusion dynamics, ensures the law of the particle system converges to the prescribed path in the continuous-time population limit, correcting for the model's score error.
3. Approximates the resulting corrected dynamics using sequential Monte Carlo, with a novel variance-controlling guidance term that prevents particle degeneracy.

---

## Technical Formulation

Let $\{p_t\}_{t=0}^T$ be the prescribed probability path (e.g., $p_0$ is the data distribution, $p_T$ is a reference Gaussian). Let $\hat{s}_\theta(x, t)$ be the pretrained score function with error $\epsilon_t(x) = \hat{s}_\theta(x,t) - \nabla \log p_t(x)$.

Standard inference-time scaling uses:
$$dx_t = \hat{s}_\theta(x_t, t)\,dt + \sqrt{2}\,dW_t$$

which tracks $p_t$ only if $\epsilon_t = 0$. EBFKC introduces a Feynman-Kac correction:
$$G_t(x) = \exp\!\left(-\int_0^t \langle \epsilon_\tau(x_\tau), \nabla \log p_\tau(x_\tau) \rangle \, d\tau\right)$$

and shows that the corrected particle system with weights proportional to $G_t$ converges to $p_t$ in the population limit regardless of $\epsilon_t$. The practical SMC approximation maintains $N$ weighted particles $\{(x_t^{(i)}, w_t^{(i)})\}_{i=1}^N$ with resampling steps governed by effective sample size monitoring, while variance-controlling guidance keeps the particle cloud from collapsing.

---

## Learning or Inference Procedure

EBFKC requires no additional training; it operates purely at inference time on a frozen pretrained diffusion model. The procedure:

1. **Specify reference energy** $E_t$: for molecular generation, this is the target free-energy functional; for reward tilting, a scalar reward model.
2. **Initialize particles** by sampling from the prior $p_T$ (typically a Gaussian).
3. **Propagate** particles forward/backward through the diffusion process using the pretrained score $\hat{s}_\theta$, computing Feynman-Kac potentials $G_t$ at each step.
4. **Reweight** particles by $G_t$; resample when effective sample size drops below a threshold.
5. **Return** the weighted particle ensemble as the approximate target distribution.

The computational overhead relative to standard SMC is a single energy evaluation per particle per step.

---

## What the Guarantee Says

In the continuous-time population limit (infinite particles, infinitely fine time discretization), EBFKC tracks the prescribed path $\{p_t\}$ exactly regardless of the model error $\epsilon_t$. Practically, error scales as $O(1/\sqrt{N})$ in particle count and $O(\Delta t)$ in discretization step size, with the Feynman-Kac correction removing the fixed systematic bias that standard SMC cannot address.

---

## Experimental Findings

- **Gaussian mixture models:** EBFKC closely matches multi-modal target distributions in 2D where standard inference-time scaling with imperfect models converges to incorrect modes.
- **Particle systems:** EBFKC accurately recovers equilibrium distributions for interacting particle systems under energy-based targets; standard baselines show persistent density overestimation of low-energy regions.
- **Alanine dipeptide (biological molecule):** EBFKC matches the ground-truth free-energy surface (Ramachandran plot) within statistical uncertainty; standard SMC baselines exhibit errors of several $k_BT$ in high-barrier regions.
- **Alanine tetrapeptide:** EBFKC scales to the larger peptide and maintains accurate free-energy profiles, demonstrating applicability to realistic biomolecular systems.

---

## Ablations and Interpretation

- **Particle count:** As $N$ increases, EBFKC converges faster and more reliably to the target than standard SMC; at low $N$, variance-controlling guidance significantly reduces particle collapse.
- **Model imperfection level:** As the pretrained model's score error $\epsilon_t$ is artificially increased (by using models trained with less data), standard SMC degrades rapidly while EBFKC is more robust.
- **Variance-controlling guidance:** Removing the variance-controlling term causes particle degeneracy, confirming its role in keeping the particle system well-spread through probability space.
- **Feynman-Kac vs. simple reweighting:** Naive importance weighting without the Feynman-Kac path correction fails to track the prescribed probability path, producing biased samples even with large $N$.

---

## Reference

Zuokai Wen, Louis Grenioux, Weinan E, Jiequn Han. **Error-Corrected Inference-Time Scaling for Imperfect Diffusion Models**. arXiv:2610.01933, October 2026.  
https://arxiv.org/abs/2610.01933
