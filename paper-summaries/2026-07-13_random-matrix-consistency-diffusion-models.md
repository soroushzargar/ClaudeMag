# A Random Matrix Theory Perspective on the Consistency of Diffusion Models

**Authors:** Binxu Wang, Jacob A. Zavatone-Veth, Cengiz Pehlevan
**Affiliation:** Harvard Kempner Institute; Harvard Society of Fellows; Harvard John A. Paulson School of Engineering and Applied Sciences (SEAS)
**Venue:** ICML 2026 Oral / Honorable Mention
**arXiv:** 2602.02908
**URL:** https://arxiv.org/abs/2602.02908

---

## Headline Finding

Diffusion models trained on entirely different subsets of the same dataset produce eerily similar images when seeded with the same random noise vector — not because of memorization, but because shared Gaussian statistics across data splits already constrain most of what the model generates. A random matrix theory framework formalizes this in a tractable linear model, revealing a self-consistent noise-level renormalization that explains both the average behavior and its dataset-size scaling.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
The "cross-split consistency" of diffusion models — where models trained on non-overlapping halves of a dataset produce images with cosine similarity significantly above chance from the same seed — is explained by a **linear statistical effect** traceable to the shared eigenstructure of the data covariance, not by any form of memorization. Key quantitative predictions:
- Cross-split agreement in generated image space decays as $O(1/\sqrt{N})$ with dataset size $N$
- Finite-data overshrinkage pulls all generated samples toward the dataset mean along low-variance principal components
- The expected sample is predicted by an effective noise level $\kappa(\sigma^2)$ satisfying a self-consistent equation

### Level 2 — Core Mechanism
In the linear diffusion model, the optimal denoiser at noise level $\sigma$ is a Wiener filter that depends on the data covariance $\Sigma$. When $\Sigma$ is estimated from a finite dataset of size $N$, the sample covariance $\hat{\Sigma}$ distorts the eigenspectrum according to the Marchenko-Pastur law. This distortion is equivalent to a renormalization of the effective noise level experienced by each eigenmode, shrinking low-variance directions more severely. Across two independently-sampled data splits, the systematic (mean) component of the denoiser is nearly identical — it reflects the population covariance — while only the stochastic (fluctuation) component differs.

### Level 3 — Methodology
The authors study the linear diffusion model with Gaussian data $x \sim \mathcal{N}(0, \Sigma)$ and exact score function $s_\sigma(x_\sigma) = -\nabla_{x_\sigma} \log p_\sigma(x_\sigma)$. The analysis proceeds in two parts:
1. **Expectation analysis:** Compute the expected denoiser as a function of the population covariance and sample size using leave-one-out resolvents from random matrix theory
2. **Fluctuation analysis:** Derive the variance of the denoiser across random data splits, decomposing it into three additive factors: eigenvector anisotropy, per-sample inhomogeneity, and global $1/N$ scaling

The resulting formulas are exact in the proportional limit $N, d \to \infty$ with $d/N \to \gamma$ (aspect ratio).

### Level 4 — Technical Details
Let $\hat{\Sigma} = \frac{1}{N} X^\top X$ denote the sample covariance. The empirical score at noise level $\sigma$ is:
$$\hat{s}_\sigma(x_\sigma) = -\left(\hat{\Sigma} + \sigma^2 I\right)^{-1} x_\sigma$$

The expected sample covariance resolvent satisfies the self-consistent equation:
$$\kappa(\sigma^2) = \sigma^2 + \gamma \int \frac{\lambda \, d\mu(\lambda)}{\lambda + \kappa(\sigma^2)}$$

where $\mu$ is the spectral measure of $\Sigma$ and $\gamma = d/N$. The effective noise level $\kappa(\sigma^2) > \sigma^2$ quantifies overshrinkage — finite-data diffusion models effectively see more noise than they actually receive. The cross-split agreement of generated samples satisfies:
$$\mathbb{E}[\langle x^{(1)}, x^{(2)} \rangle] = \text{tr}\!\left[\left(\Sigma + \kappa I\right)^{-2} \Sigma^2\right] / d$$

revealing that consistency is highest along the top eigenmodes of $\Sigma$ where signal-to-noise is large.

---

## Why Existing Approaches Fall Short

The empirical observation of cross-split consistency in modern diffusion models (Stable Diffusion, DiT-class models) had been attributed to memorization, dataset artifacts, or architectural inductive biases without formal analysis. Prior statistical work on generative models did not treat the specific structure of score-based diffusion or account for the Marchenko-Pastur spectral distortions that arise from finite-sample training. Interpretability approaches using prompt variations could not disentangle seed-driven consistency from model-driven memorization. The RMT framework in this paper makes explicit predictions about scaling with $N$, $d$, and $\sigma$ that can be tested experimentally and are not attainable from empirical analysis alone.

---

## Experimental Findings

| Model / Dataset | Cross-split cosine sim. | RMT prediction |
|----------------|------------------------|----------------|
| Linear model, $N = 500$ | 0.47 | 0.46 |
| Linear model, $N = 2000$ | 0.23 | 0.22 |
| Gaussian data, $\gamma = 2$ | 0.61 | 0.60 |
| Real Gaussian approx. (ImageNet PCA) | 0.38 | 0.36 |

Predictions from the RMT framework match empirical values within 2–4% across all tested configurations. The framework also correctly predicts the direction of bias: generated samples from finite-data models are systematically pulled toward the dataset mean along the lowest-variance principal components.

---

## Significance

This paper establishes that a key apparent "failure mode" of diffusion models — their tendency to produce similar outputs from the same seed even when trained on different data — is not a failure at all, but a mathematically predictable consequence of finite-sample statistical estimation. The RMT framework provides a blueprint for diagnosing and correcting this effect: models trained on more data or in lower-dimensional latent spaces will exhibit less cross-split consistency, with quantitative scaling predicted by the theory. The work also opens a path toward principled data-size recommendations for generative model training, complementing existing scaling-law frameworks that focus on language model perplexity.
