# On the Diffusibility of High-Dimensional Latents

**arXiv:** 2609.28473  
**Authors:** Chao Feng, Zhiyang Xu, Bowei Chen, Yuanjun Xiong, Xiyao Wang, Jui-Hsien Wang, Richard Zhang, Zhe Lin, Andrew Owens, Yijun Li  
**Date:** September 2026  
**Venue:** ECCV 2026

---

## One-Line Summary

Reconstruction-finetuned visual encoders collapse to a low-dimensional signal manifold, making standard velocity prediction inefficient in high-dimensional flow matching; switching to clean-data (x₀) prediction focuses optimization on the signal subspace and substantially improves generation quality.

---

## Problem

Latent diffusion models operate in the compressed latent spaces of autoencoders rather than pixel space, enabling efficient training and generation. A natural extension is to use the feature spaces of powerful pretrained visual encoders (CLIP, DINOv2, etc.) as the latent space, leveraging their rich semantic structure. Representation Autoencoders (RAEs) add a decoder to these encoders to enable reconstruction. However, many off-the-shelf encoders discard fine-grained visual details (they are trained for classification, not reconstruction), so RAEs must be finetuned for reconstruction quality.

This finetuning introduces a subtle geometric problem: while the nominal feature dimension remains large (e.g., 768 or 1024), the learned features concentrate near a much lower-dimensional subspace. Standard flow matching with velocity prediction must fit orthogonal noise directions outside this low-dimensional signal manifold, making optimization inefficient and generation quality poor.

---

## Why Existing Approaches Fall Short

- **Standard pixel-space VAE latents** (used in Stable Diffusion) are designed for diffusion from the ground up, with controlled dimensionality and well-conditioned geometry; RAE latents have neither property by default.
- **Velocity prediction (v-prediction / ε-prediction)** in flow matching assumes the data manifold fills the ambient space; when data concentrates on a low-dimensional subspace, the velocity field has large orthogonal components that carry no signal and harm optimization.
- **Higher feature dimensions** (for richer representations) make this problem worse, not better.
- **Dimensionality reduction** (PCA projection before diffusion) throws away potentially useful representation structure and requires additional engineering.

---

## Core Method

**Effective Dimensionality Analysis.** The paper first characterizes the geometry of reconstruction-finetuned encoder features by computing the effective dimensionality $d_\text{eff}$ via the participation ratio:
$$d_\text{eff} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}$$
where $\lambda_i$ are eigenvalues of the feature covariance matrix. RAE features show a sharp collapse: $d_\text{eff} \ll d_\text{nominal}$.

**x₀-Prediction Instead of Velocity Prediction.** The paper proposes a simple fix: train the flow-matching model to predict the clean data $\mathbf{x}_0$ directly, rather than the velocity field $\mathbf{v} = \mathbf{x}_0 - \boldsymbol{\epsilon}$.

In standard flow matching (linear interpolation):
$$\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon}, \quad t \in [0, 1]$$

Velocity target: $\mathbf{v} = \mathbf{x}_0 - \boldsymbol{\epsilon}$ (has large orthogonal components when $d_\text{eff} \ll d$).

Data target: $\mathbf{x}_0$ (concentrates on the signal manifold, avoids fitting orthogonal noise directions).

**Why This Helps.** The model fitting error for orthogonal directions is eliminated when predicting $\mathbf{x}_0$: since $\mathbf{x}_0$ itself lies on the low-dimensional signal manifold, the optimization landscape for $\mathbf{x}_0$-prediction is lower-dimensional and better-conditioned than for velocity prediction.

---

## Technical Formulation

Let $\mathbf{x}_0 \in \mathbb{R}^d$ be a RAE feature (with effective dimensionality $d_\text{eff} \ll d$), and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ be isotropic noise.

The noisy sample at time $t$:
$$\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon}$$

Velocity-prediction training loss:
$$\mathcal{L}_v = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|f_\theta(\mathbf{x}_t, t) - (\mathbf{x}_0 - \boldsymbol{\epsilon})\|^2\right]$$

Data-prediction training loss:
$$\mathcal{L}_{x_0} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|f_\theta(\mathbf{x}_t, t) - \mathbf{x}_0\|^2\right]$$

The key insight: the velocity target $\mathbf{x}_0 - \boldsymbol{\epsilon}$ has noise components $-\boldsymbol{\epsilon}$ that span the full $d$-dimensional space, whereas the data target $\mathbf{x}_0$ lies on the $d_\text{eff}$-dimensional signal manifold. Fitting the full $d$-dimensional velocity is harder when $d_\text{eff} \ll d$.

---

## Learning or Inference Procedure

**Training:**
1. Finetune the pretrained visual encoder as a RAE for reconstruction quality.
2. Extract RAE features for the training image set.
3. Compute $d_\text{eff}$ to characterize the signal manifold.
4. Train a flow-matching generator with $\mathcal{L}_{x_0}$ instead of $\mathcal{L}_v$.

**Inference:**
1. Sample $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$.
2. Numerically integrate the learned flow from $t=T$ to $t=0$ using the $\mathbf{x}_0$ prediction.
3. Decode the generated feature $\mathbf{x}_0$ with the RAE decoder to obtain an image.

---

## What the Paper Claims

1. **Effective dimensionality collapse is systematic:** Reconstruction finetuning consistently reduces $d_\text{eff}$ to a small fraction of $d_\text{nominal}$ across encoder architectures.
2. **Velocity prediction is inefficient in collapsed spaces:** The orthogonal noise components in the velocity target explain poor generation quality in high-dimensional RAE spaces.
3. **x₀-prediction substantially improves generation:** Switching parameterization substantially improves FID and CLIP scores without any other architectural change.

---

## Experimental Findings

- Evaluated with multiple pretrained encoder families (CLIP ViT-L/14, DINOv2) at various feature dimensions.
- Effective dimensionality $d_\text{eff}$ is 5–15% of nominal dimensionality after reconstruction finetuning.
- $\mathbf{x}_0$-prediction improves FID by a substantial margin (exact numbers in the ECCV paper) compared to velocity prediction in the same RAE latent space.
- CLIP similarity scores also improve, confirming that the benefit is not purely statistical.
- Results are robust across encoder types and RAE decoder architectures.

---

## Ablations and Interpretation

- **Nominal vs. effective dimensionality:** The FID gap between $\mathbf{x}_0$- and $v$-prediction correlates with $d/d_\text{eff}$, directly supporting the manifold mismatch hypothesis.
- **No RAE finetuning (frozen encoder):** When the encoder is not finetuned for reconstruction, $d_\text{eff}$ is higher and the gap between parameterizations is smaller.
- **Alternative fixes (PCA projection):** Projecting to the principal subspace before velocity prediction partially recovers quality but requires separate PCA fitting and discards features.

---

## Reference

Chao Feng, Zhiyang Xu, Bowei Chen, Yuanjun Xiong, Xiyao Wang, Jui-Hsien Wang, Richard Zhang, Zhe Lin, Andrew Owens, Yijun Li. **On the Diffusibility of High-Dimensional Latents**. arXiv:2609.28473, ECCV 2026.  
https://arxiv.org/abs/2609.28473
