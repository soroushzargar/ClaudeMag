# Rethinking the Design Space of Vanilla Pixel Diffusion Transformers

**Authors:** Soumik Mukhopadhyay, Long Mai, Abhinav Shrivastava, Aniruddha Mahapatra  
**Venue:** arXiv preprint (under review)  
**arXiv:** 2610.23854  
**Date:** October 16, 2026  
**URL:** https://arxiv.org/abs/2610.23854

---

## Summary

Pixel-space diffusion transformers (DiTs) generate images directly in pixel space rather than in the compressed latent space used by most modern image generators. While latent diffusion models dominate in quality, pixel DiTs offer conceptual simplicity, architectural transparency, and avoid the distortions introduced by VAE tokenization. This paper revisits a key design choice in pixel DiTs—the prediction parameterization—and proposes MFlow, an adaptive interpolation between clean-image (x₀) and velocity (u) prediction that is learned jointly with the diffusion model.

---

## Background: Prediction Parameterization

A diffusion model predicts a quantity from a noisy observation x_t:

- **x₀-prediction:** Directly predict the clean image.
- **ε-prediction:** Predict the noise added to the clean image.
- **u-prediction (velocity):** Predict a linear combination of x₀ and noise, i.e., u = αx₀ − σε.

The choice of parameterization affects gradient magnitudes, convergence speed, and final sample quality in a way that depends on the noise schedule, model capacity, and data characteristics. Prior work (JiT, AsymFlow) has shown that x₀-prediction can outperform ε-prediction for pixel DiTs, but the optimal choice varies across settings.

---

## MFlow: Adaptive Parameterization

MFlow introduces a per-principal-direction gain m_i ∈ [0, 1] that interpolates between x₀- and u-prediction for each data direction:

**target_i = m_i · u_i + (1 − m_i) · (x₀)_i**

When m_i = 0, the model predicts clean pixels for direction i; when m_i = 1, it predicts velocity. The gains are initialized from empirical signal-to-noise statistics and updated end-to-end during diffusion training.

**Key finding:** Higher-capacity models systematically prefer velocity prediction (larger m_i), while lower-capacity models prefer clean-image prediction. MFlow discovers this preference automatically.

---

## ConvRefiner: Addressing High-Frequency Deficits

All vanilla pixel DiTs—regardless of parameterization—retain a persistent high-frequency deficit: generated images lack fine texture details compared to real images. To address this, the paper adds a lightweight convolutional module (ConvRefiner) trained post-hoc with an adversarial loss while keeping the diffusion backbone frozen. ConvRefiner recovers high-frequency detail with minimal additional compute.

---

## Key Results

- **ImageNet (class-conditional):** MFlow achieves FID scores competitive with or better than fixed-parameterization baselines at each model scale.
- **UCF-101 (video generation):** MFlow transfers to pixel-space video generation, outperforming x₀- and u-only baselines.
- **Large-scale text-to-image:** MFlow maintains consistent gains at 1B+ parameter scales.
- **AsymFlow comparison:** The 1.57 FID of the rank-asymmetric AsymFlow baseline (a separate work) is matched or exceeded by MFlow + ConvRefiner on the same architecture.

---

## Significance

MFlow provides a principled and practical solution to the parameterization dilemma in pixel diffusion transformers. By treating prediction parameterization as a trainable quantity rather than a fixed hyperparameter, the method removes a key impediment to pixel DiT adoption and narrows the quality gap with latent diffusion models.

---

## Citation

Soumik Mukhopadhyay, Long Mai, Abhinav Shrivastava, Aniruddha Mahapatra.  
**Rethinking the Design Space of Vanilla Pixel Diffusion Transformers.**  
arXiv:2610.23854. Under review, 2026.  
https://arxiv.org/abs/2610.23854
