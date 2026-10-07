# S2PD: Serial-to-Parallel Diffusion for Physically and Logically Consistent Video Generation

**arXiv:** 2610.06847  
**Authors:** Jeffrey Hu, Daniel Olmeda Reino, Ayush Tewari  
**Affiliation:** Massachusetts Institute of Technology  
**Date:** October 5, 2026

---

## One-Line Summary

S2PD resolves the "seriality gap" in video diffusion by running autoregressive diffusion at high noise—to coordinate causally dependent events—then switching to parallel diffusion at low noise for efficient joint refinement.

---

## Problem

Modern video diffusion models denoise entire video clips in parallel, achieving state-of-the-art visual quality. However, a persistent problem—the **seriality gap**—has been documented: bidirectional diffusion models systematically fail to respect physical laws and logical consistency rules even when trained to convergence on unlimited in-distribution data from procedural generators. Collision outcomes, causal chains, and sequential logical constraints are routinely violated, regardless of how many denoising steps are allocated.

---

## Why Existing Approaches Fall Short

- **The seriality gap is structural, not data-driven:** The failure of bidirectional diffusion on causally dependent events persists even with unlimited in-distribution training data and longer denoising schedules, indicating a fundamental architectural limitation.
- **More denoising steps do not help:** For deterministic video prediction tasks, increasing the number of denoising steps does not add effective serial computation beyond the backbone depth, since each denoising step still processes the entire video simultaneously.
- **Autoregressive generation is serial but slow:** Fully autoregressive video generation adds the serial computation needed for causal coordination but generates one frame at a time, making sampling prohibitively slow for high-resolution or long videos.
- **No prior method combines both advantages:** Existing work treats parallelism and serialism as opposed choices, missing the insight that different noise levels benefit from different computational strategies.

---

## Core Method

**S2PD (Serial-to-Parallel Diffusion)** is a hybrid inference algorithm with two phases:

1. **Serial (autoregressive) phase at high noise:** At noise levels above a threshold $\sigma^*$, the video is denoised autoregressively—one frame (or segment) at a time, with each frame conditioned on the already-denoised preceding frames. This provides the serial compute needed to correctly propagate causal dependencies: collision outcomes, logical rule sequences, and multi-step physical processes.

2. **Parallel phase at low noise:** Once the high-level structure and causal skeleton are established (i.e., the denoising trajectory reaches noise level below $\sigma^*$), the model switches to full parallel denoising over the entire video. This jointly refines visual detail and reduces total sampling time relative to a fully serial approach.

The switching point $\sigma^*$ is a hyperparameter set based on the noise schedule; empirically, switching at the midpoint of the noise schedule provides a good balance.

---

## Technical Formulation

Let $\mathbf{x}_0$ denote the clean video of $T$ frames, $\mathbf{x}_{\sigma}$ the noisy video at noise level $\sigma$, and $\epsilon_\theta$ the learned denoiser. Standard parallel diffusion denoises all frames simultaneously:
$$\hat{\mathbf{x}}_0 = \epsilon_\theta(\mathbf{x}_\sigma, \sigma)$$

S2PD decomposes the denoising trajectory at threshold $\sigma^*$. For $\sigma > \sigma^*$ (high noise), it uses autoregressive generation:
$$\hat{x}_{0,t} = \epsilon_\theta(x_{\sigma,t}, \sigma \mid x_{0,1:t-1})$$

for $t = 1, \ldots, T$ sequentially, where conditioning on $x_{0,1:t-1}$ is implemented via cross-attention or masked attention.

For $\sigma \leq \sigma^*$ (low noise), it switches to parallel denoising:
$$(\hat{x}_{0,1}, \ldots, \hat{x}_{0,T}) = \epsilon_\theta(\mathbf{x}_\sigma, \sigma)$$

starting from the video latent obtained at the end of the serial phase. Total sampling time is approximately $(T-1) \cdot t_\text{serial} + t_\text{parallel}$, substantially less than $T \cdot t_\text{serial}$ for fully autoregressive generation.

---

## Learning or Inference Procedure

**No new training is required:** S2PD is an inference-time algorithm applied to an existing pretrained bidirectional diffusion model that supports autoregressive conditioning (either via masked attention or a separate autoregressive head). The switching threshold $\sigma^*$ is set as a hyperparameter.

**At inference:**
1. Sample a Gaussian noise video $\mathbf{x}_{\sigma_\text{max}}$.
2. Run the standard DDIM/DPM solver serially (frame by frame) from $\sigma_\text{max}$ down to $\sigma^*$.
3. From $\sigma^*$ down to $0$, run the solver in parallel over all frames.
4. Return the denoised video $\hat{\mathbf{x}}_0$.

---

## What the Guarantee Says

The paper does not provide a formal statistical guarantee. The claim is empirical: on benchmarks requiring physical and logical consistency (procedurally generated game environments, simulated physics tasks, and real video), S2PD is more consistent than bidirectional baselines while requiring less sampling time than fully autoregressive methods. The intuition is that the high-noise phase determines the global causal structure of the video, so resolving it serially is both necessary and sufficient for consistency.

---

## Experimental Findings

- **Games:** S2PD follows game rules (e.g., Tetris collision physics, Sokoban push logic) significantly more reliably than bidirectional baselines trained on the same data.
- **Physical simulations:** On billiards-style collision tasks and rigid-body interactions, S2PD achieves lower rule-violation rates.
- **Real video:** On real-world video prediction benchmarks, S2PD improves temporal stability metrics while maintaining visual quality.
- **Sampling efficiency:** S2PD is substantially faster than fully autoregressive generation while closing the consistency gap with serial methods.

---

## Ablations and Interpretation

**Effect of switching threshold $\sigma^*$:** Moving $\sigma^*$ earlier (less serial computation) degrades consistency; moving it later increases consistency but reduces the sampling efficiency advantage. The midpoint is Pareto-optimal.

**Ablation on serial vs.\ parallel phases:** Removing the parallel phase (fully serial) matches consistency but increases sampling time by $T\times$; removing the serial phase (fully parallel) reduces sampling time but restores the seriality gap.

**Error analysis:** Violations are concentrated in frames that require coordination of state from distant prior frames, confirming that the seriality gap is a long-range causal dependency problem rather than a local temporal issue.

---

## Reference

Jeffrey Hu, Daniel Olmeda Reino, Ayush Tewari.
**S2PD: Serial-to-Parallel Diffusion for Physically and Logically Consistent Video Generation.**
arXiv:2610.06847, October 2026.
https://arxiv.org/abs/2610.06847
