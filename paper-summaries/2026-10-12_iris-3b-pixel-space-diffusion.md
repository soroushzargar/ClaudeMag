# Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning

**arXiv:** 2610.09450  
**Authors:** Hanqiu Li Cai, Chema Garabito  
**Affiliation:** Speridlabs  
**Date:** October 7, 2026

---

## One-Line Summary

Iris-3B is a 3-billion-parameter text-to-image diffusion model trained entirely in pixel space—without a VAE encoder—via a 256→512→1024 resolution curriculum; it matches latent diffusion baselines on generation quality but finds no advantage on fine-grained downstream tasks, suggesting the VAE bottleneck is less limiting than assumed.

---

## Problem

The dominant paradigm in high-resolution text-to-image generation (Stable Diffusion, FLUX, etc.) uses a **Variational Autoencoder (VAE)** to compress images into a low-dimensional latent space before applying diffusion. This reduces computation but introduces VAE reconstruction artefacts that may harm pixel-level tasks like depth estimation and image restoration. The hypothesis this paper tests is: if the VAE is the bottleneck, training diffusion directly in pixel space should yield a measurable advantage on such tasks.

---

## Why Existing Approaches Fall Short

- **Latent diffusion models** (LDMs) encode images into 4–16× downsampled latent codes; the VAE decoder reconstructs from these, sometimes blurring fine detail or introducing checkerboard artefacts.
- **Pixel-space diffusion at high resolution** is computationally expensive: the UNet or transformer must process full-resolution feature maps, which scale quadratically in spatial dimensions.
- Previous pixel-space models (early DALL-E, Imagen) used cascaded super-resolution but operated at lower resolutions or required separate upsampling stages.
- **Converting a pre-trained latent model to pixel space** is an unexplored alternative to training from scratch.

---

## Core Method

Iris-3B takes two parallel approaches to pixel-space diffusion:

**Approach 1 — Train from scratch:**
- A 3B-parameter pixel-space diffusion transformer (PiT head from PixelDiT architecture) is trained on a large-scale image–caption dataset.
- A **progressive resolution curriculum** trains the model at 256² first, then 512², then 1024², allowing the model to learn low-frequency structure before high-frequency detail.
- Text conditioning uses Qwen3-VL-4B-Instruct as the text encoder.

**Approach 2 — Convert from latent:**
- FLUX.2 Klein 4B (a pre-trained latent diffusion model) is converted to pixel space by replacing the VAE encoder/decoder with an identity mapping and adapting the channel dimensions.
- The converted model is fine-tuned on the same pixel-space objective.

Both variants are then fine-tuned on downstream tasks (monocular depth estimation, 4× image super-resolution).

---

## Technical Formulation

The diffusion objective in pixel space is the standard denoising score matching:
$$\mathcal{L}_\text{DSM} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})\|_2^2\right]$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$ is the noisy image at timestep $t$, $\mathbf{c}$ is the text embedding, and $\boldsymbol{\epsilon}_\theta$ is the pixel-space denoising network.

For fine-tuning on depth estimation with ground-truth depth maps $\mathbf{d}$:
$$\mathcal{L}_\text{depth} = \mathcal{L}_\text{DSM}(\mathbf{d}) + \lambda \|\hat{\mathbf{d}} - \mathbf{d}\|_1$$

The progressive curriculum uses a staged training schedule $\{(256^2, T_1), (512^2, T_2), (1024^2, T_3)\}$ with the number of training steps $T_i$ decreasing as resolution increases.

---

## Learning or Inference Procedure

**Pre-training:**
1. Stage 1: train at 256² for $T_1$ steps on image–caption pairs.
2. Stage 2: upsample spatial dimensions, continue at 512² for $T_2$ steps.
3. Stage 3: continue at 1024² for $T_3$ steps.

**Downstream fine-tuning:**
1. Replace the text-conditioning input with the appropriate conditioning signal (e.g., an image for super-resolution or depth estimation).
2. Fine-tune with a task-specific loss for a small number of steps.

**Inference:**
- Standard DDPM/DDIM sampling in pixel space; no VAE decoding step.

---

## What the Guarantee Says

No formal guarantees are claimed. The paper makes empirical claims on GenEval and DPG-Bench benchmarks. The main empirical finding is *negative*: despite operating without a VAE, Iris-3B does not outperform latent diffusion baselines on pixel-level tasks.

---

## Experimental Findings

- **Generation quality:** Iris-3B achieves **GenEval 0.798** and **DPG-Bench 86.52** at 1024², matching Qwen-Image and competitive with latent FLUX.2 Klein.
- **Monocular depth estimation:** Iris-3B performs at parity with a latent FLUX.2 Klein fine-tune; the converted pixel model performs *worse*.
- **4× super-resolution (DIV2K):** Neither Iris-3B nor the converted model outperforms the latent FLUX.2 Klein fine-tune.
- **Inference compute:** Pixel-space inference at 1024² is substantially more expensive per step than latent inference at equivalent visual quality.

---

## Ablations and Interpretation

**Curriculum stages:** Skipping the 256² stage degrades final generation quality; the low-frequency pre-training is important for convergence at high resolution.

**Conversion quality:** The converted pixel model underperforms the from-scratch model and the original latent model, suggesting that the VAE's representation is not easily transferred.

**Text encoder:** Replacing Qwen3-VL-4B with a smaller text encoder degrades DPG alignment score more than GenEval object count, indicating that text alignment is particularly encoder-sensitive.

**Interpretation:** The null result on downstream tasks suggests either (a) the VAE bottleneck is less harmful than assumed, (b) the latent representation learned by FLUX-style models is beneficial for downstream tasks beyond generation, or (c) the pixel-space model needs more task-specific fine-tuning data to close the gap.

---

## Reference

Hanqiu Li Cai, Chema Garabito (Speridlabs).
**Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning.**
arXiv:2610.09450, October 2026.
https://arxiv.org/abs/2610.09450
