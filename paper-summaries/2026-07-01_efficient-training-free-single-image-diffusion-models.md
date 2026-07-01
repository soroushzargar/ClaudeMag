# Efficient and Training-Free Single-Image Diffusion Models

### Patch Statistics Replace Neural Networks for Single-Image Generation at CVPR 2026

**Authors:** Haojun Qiu, Kiriakos N. Kutulakos, David B. Lindell (University of Toronto / Vector Institute)  
**Venue:** CVPR 2026 Highlight  
**arXiv:** [2606.04299](https://arxiv.org/abs/2606.04299)  
**Submitted:** June 2026  
**Project page:** https://haojunqiu.github.io/efficient-SID/

---

## Layer 1 — The Big Picture

Single-image generation (SIG) — learning the internal distribution of a single natural image and sampling novel images from it — is a fundamental problem in image synthesis and manipulation. It enables unconditional generation, guided image editing, texture synthesis, style transfer, and retargeting from a single reference.

Prior approaches train a small diffusion model on patches extracted from the single reference image. While powerful, this requires hours of GPU training per image — a steep cost for a task that in principle needs only the reference statistics.

This paper proposes a fundamentally different approach: instead of training any neural network, the score function for a noisy image patch is computed analytically in closed form using the empirical patch distribution of the single image. This yields a **training-free diffusion sampler** that runs orders of magnitude faster than trained alternatives while achieving or surpassing their generation quality.

The method exploits the observation that single images have only ~10,000 patches at each scale, making the empirical patch distribution a tractable and sufficient description of the image's internal structure. Combined with a multi-scale hierarchical sampling scheme, the method supports unconditional generation, text-guided stylization, image symmetrization, and retargeting.

---

## Layer 2 — Under the Hood

**Why training is unnecessary for single images.**

For a large natural image dataset, the true score function $\nabla_x \log p(x)$ is intractable — there are exponentially many images, so one must approximate it with a neural network (a denoising autoencoder). But for a single image, the internal patch distribution is finite: a $512 \times 512$ image has roughly $256 \times 256 = 65,536$ distinct $8 \times 8$ patches. This distribution is discrete and fully enumerable.

The key insight: the denoising score function for a patch $y = x + \epsilon$ (where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$) is exactly the posterior mean $\mathbb{E}[x | y]$ of the clean patch. When the prior over clean patches is the empirical distribution (a sum of Dirac deltas at each observed patch), this posterior mean has a closed-form solution via Nadaraya-Watson kernel smoothing:

$$\mathbb{E}[x | y] = \frac{\sum_{i} x_i \cdot \exp\left(-\frac{\|y - x_i\|^2}{2\sigma^2}\right)}{\sum_{i} \exp\left(-\frac{\|y - x_i\|^2}{2\sigma^2}\right)}$$

where $\{x_i\}$ is the set of all patches from the reference image. This is a Gaussian kernel regression over the patch set — cheap to compute and provably optimal under the empirical patch prior.

**Multi-scale architecture.** The single-level denoiser captures per-patch statistics but not spatial coherence. The authors introduce a coarse-to-fine hierarchical scheme: generation proceeds from a low-resolution grid (capturing global layout) down to full resolution (capturing fine textures). At each scale, the training-free patch denoiser is applied, conditioned on the upsampled coarse result.

---

## Layer 3 — The Math

**Score function derivation.** Define the patch dataset $\mathcal{P} = \{x_1, \ldots, x_N\}$ from the reference image $I$. For a noisy patch $y = x^* + \sigma\epsilon$, $\epsilon \sim \mathcal{N}(0,I)$, the optimal denoiser under mean squared error is:

$$D_\sigma(y) = \mathbb{E}[x | y] = \sum_{i=1}^{N} w_i(y, \sigma) \cdot x_i$$

with weights:
$$w_i(y, \sigma) = \frac{\exp\left(-\frac{\|y - x_i\|^2}{2\sigma^2}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{\|y - x_j\|^2}{2\sigma^2}\right)}$$

The score function is then:
$$\nabla_y \log p_\sigma(y) = \frac{D_\sigma(y) - y}{\sigma^2}$$

This is the Tweedie formula, applied to the empirical patch prior.

**Efficient computation.** With $N$ patches of dimension $d$ (e.g., $N = 10^4$, $d = 64$ for $8\times8$ RGB patches), each denoising step requires computing $N$ inner products — $O(Nd)$ operations. This is much cheaper than a neural network forward pass for the same image size.

**Multi-scale sampling.** At resolution level $l \in \{1, \ldots, L\}$, generate patches $\{x^{(l)}_j\}$ conditioned on the upsampled output from level $l-1$. The conditioning is implemented via classifier-free-style guidance: the noisy patch denoiser at level $l$ is modified to also condition on the coarse-level reconstruction.

---

## Experiments & Datasets

**Baselines:** SinDDPM (trained single-image diffusion), SSGAN, ConSinGAN, SinFusion.

**Tasks:** Unconditional random generation, text-guided stylization, image symmetrization, image retargeting.

**Results:** On standard single-image benchmarks (SIFID, FID on generated samples), the training-free method achieves lower SIFID than SinDDPM on most textures, while running **300× faster** (seconds vs. hours). Text-guided stylization produces coherent results without any fine-tuning. Accepted as a CVPR 2026 Highlight.
