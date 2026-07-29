# DiFA: Inference-Time Forward-Process Alignment for Diffusion Models

**Authors:** Shigui Li, Delu Zeng  
**Affiliation:** South China University of Technology  
**Venue:** arXiv:2607.17972 (ICML 2026)  
**Date:** July 19, 2026  
**URL:** https://arxiv.org/abs/2607.17972

---

## Summary

Standard diffusion model inference treats the reverse process as a deterministic numerical integration problem, applying each denoising step independently. DiFA (Diffusion Forward-process Alignment) reframes inference as a sequential state estimation problem, using Kalman-filter-inspired temporal consensus across denoising steps to leverage the statistical structure of the forward process. The result is a training-free wrapper that significantly improves FID, IS, and FD-DINOv2 on CIFAR-10 and ImageNet when wrapped around any existing sampler.

## Problem

Diffusion models are trained to reverse a forward noising process $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$. At inference, the reverse step predicts $\hat{x}_0^{(t)} = f_\theta(x_t, t)$ and uses it to compute $x_{t-1}$. This formulation treats each prediction $\hat{x}_0^{(t)}$ as an independent, exact estimate of $x_0$—ignoring the statistical relationship between estimates at different time steps.

In practice, early predictions (large $t$) are coarse but structurally consistent with the forward process; later predictions are finer but may drift in ways inconsistent with the forward statistics. Treating each step independently discards information about how predictions at nearby time steps should relate to each other.

## Method: DiFA

DiFA introduces **Forward-Process Alignment (FPA)**: a temporal consensus mechanism that aggregates the history of predictions $\{\hat{x}_0^{(T)}, \hat{x}_0^{(T-1)}, \ldots, \hat{x}_0^{(t)}\}$ into a single aligned estimate $\tilde{x}_0^{(t)}$.

The consensus is computed as a noise-level-weighted average of historical predictions, where the weight of prediction $\hat{x}_0^{(\tau)}$ at current step $t$ is:

$$w(\tau, t) \propto \exp\!\left(-\frac{\|\hat{x}_0^{(\tau)} - \hat{x}_0^{(t)}\|^2}{2\sigma^2(\tau, t)}\right) \cdot \frac{1-\bar\alpha_\tau}{1-\bar\alpha_t}$$

The first term down-weights predictions that are structurally inconsistent with the current prediction; the second term scales by the noise-level ratio, giving higher weight to predictions made at similar noise levels.

To counteract the over-smoothing tendency of temporal averaging, DiFA adds **Deviation Guidance**: a perturbation in the direction of $\hat{x}_0^{(t)} - \tilde{x}_0^{(t)}$ scaled by a guidance coefficient $\gamma$, preserving fine-grained details that consensus would otherwise wash out.

## Technical Formulation

The final DiFA prediction at step $t$ is:

$$\hat{x}_0^\text{DiFA}(t) = \tilde{x}_0^{(t)} + \gamma \cdot (\hat{x}_0^{(t)} - \tilde{x}_0^{(t)})$$

where $\gamma \in [0, 1]$ controls the deviation guidance strength. When $\gamma = 0$, the output is pure temporal consensus; when $\gamma = 1$, DiFA reduces to the standard independent-step estimator.

The consensus estimate $\tilde{x}_0^{(t)}$ is derived from a Kalman-filter update analogous to fusing observations $\hat{x}_0^{(\tau)}$ with a state prior defined by the forward process statistics:

$$\tilde{x}_0^{(t)} = \sum_{\tau \geq t} w(\tau, t) \hat{x}_0^{(\tau)}, \quad \sum_{\tau} w(\tau, t) = 1$$

## Theoretical Motivation

The Kalman-filter analogy is exact when predictions are Gaussian: the forward process defines a state prior, and each prediction $\hat{x}_0^{(\tau)}$ is a noisy observation of $x_0$ with noise variance proportional to $(1-\bar\alpha_\tau)$. Optimal linear fusion then gives the minimum-variance estimator, which takes the form of the noise-level-weighted average above.

DiFA approximates this optimal estimator in the non-Gaussian case by replacing the theoretical noise covariance with the empirical prediction discrepancy $\|\hat{x}_0^{(\tau)} - \hat{x}_0^{(t)}\|^2$.

## Key Results

DiFA is evaluated wrapping three standard samplers: DDIM, DPM-Solver++, and EDM.

**CIFAR-10 unconditional (FID ↓):**
| Sampler | Baseline | + DiFA | Improvement |
|---------|---------|--------|-------------|
| DDIM (20 steps) | 4.67 | 3.21 | −1.46 |
| DPM-Solver++ (20 steps) | 3.44 | 2.87 | −0.57 |
| EDM (35 steps) | 2.04 | 1.78 | −0.26 |

**ImageNet 256×256 class-conditional (FID ↓):**
| Sampler | Baseline | + DiFA |
|---------|---------|--------|
| DDIM (50 steps) | 12.4 | 8.9 |
| DPM-Solver++ (50 steps) | 7.8 | 5.6 |

FD-DINOv2 (perceptual quality) shows consistent improvements across all configurations.

## Ablation

- Setting $\gamma = 0$ (pure consensus): FID improves but visual sharpness drops.
- Setting $\gamma = 1$ (no consensus): equivalent to baseline, confirming the consensus is doing the work.
- Using only last-step history vs. full history: using full history improves FID by an additional 0.3–0.8 points.
- Replacing noise-level weighting with uniform averaging: FID degrades by 0.4–1.2 points, confirming that noise-level compatibility is essential.

## Significance

DiFA is the first to reframe diffusion inference as sequential state estimation, importing the Kalman-filter machinery into sampling-time computation. It requires no additional training, no model modifications, and wraps any existing sampler—making it an immediate plug-in improvement for deployed diffusion systems.
