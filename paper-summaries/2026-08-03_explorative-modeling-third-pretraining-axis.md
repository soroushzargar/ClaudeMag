# Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation

**Authors:** Alexi Gladstone, Heng Ji, Yilun Du  
**Affiliation:** University of Illinois Urbana-Champaign; MIT  
**Venue:** arXiv:2607.27372  
**Date:** July 29, 2026  
**URL:** https://arxiv.org/abs/2607.27372

---

## Summary

Standard generative model training computes a loss over all data points and updates the model to reduce the expected loss. This encourages predictions to hedge across modes of the data distribution rather than committing to any one—a form of mode averaging that produces blurry or incoherent generations. This paper introduces Explorative Modeling (XM), a paradigm that treats exploration as a first-class pretraining axis and shows it yields dramatic efficiency gains without changes to architectures or data.

## Problem

The core tension in generative modeling is between coverage and commitment. Maximum likelihood training penalizes any generation that does not cover all modes of the training distribution, which pushes models to average over modes rather than select one. This is especially harmful in multimodal distributions (images with multiple valid completions, videos with multiple plausible continuations, text with multiple valid phrasings). Existing remedies—guidance, fine-tuning, best-of-N sampling at inference—address mode commitment at test time but leave the fundamental training loss unchanged.

## Method: Explorative Modeling

Explorative Modeling factors the training loop itself. Rather than computing a loss from a single model generation matched to a single training sample, XM:

1. Samples $K$ candidate generations from the current model for each training example.
2. Computes the best match between the $K$ candidates and the training target (using a per-task similarity metric such as perceptual loss for images or token overlap for text).
3. Computes the training gradient only on the best-matching candidate.

This "explore then commit" approach ensures that gradients push the model toward its own best-case output, not toward a blurred average. The training loop naturally finds modes of the distribution, because the selected candidate already represents a mode commitment. The cost is $K$ forward passes per training step, but because only one backward pass is taken, the per-step compute increase is sublinear in $K$.

## Technical Formulation

Let $p_\theta(x | c)$ be the generative model with parameters $\theta$, conditioned on context $c$. Standard MLE training minimizes:

$$\mathcal{L}_{\text{MLE}}(\theta) = -\mathbb{E}_{(x, c) \sim \mathcal{D}}\left[\log p_\theta(x | c)\right]$$

Explorative Modeling replaces this with:

$$\mathcal{L}_{\text{XM}}(\theta) = -\mathbb{E}_{(x, c) \sim \mathcal{D}}\left[\log p_\theta\!\left(\hat{x}^* | c\right)\right]$$

where $\hat{x}^* = \arg\max_{\hat{x} \in \{\hat{x}_1,\ldots,\hat{x}_K\}} \mathrm{sim}(\hat{x}, x)$ is the best-matching candidate from $K$ samples $\hat{x}_1,\ldots,\hat{x}_K \sim p_\theta(\cdot | c)$.

The similarity function $\mathrm{sim}(\cdot, \cdot)$ is task-dependent: perceptual distance for images, video quality metrics for video, and token-level ROUGE or BPE overlap for language. The gradient with respect to $\theta$ is computed only on the selected $\hat{x}^*$; the other $K-1$ candidates are discarded.

## Learning Procedure

XM requires no changes to the model architecture, optimizer, or data pipeline. The only change is the training loop:

1. For each batch, generate $K$ candidates per example.
2. Score all candidates against the ground truth.
3. Retain only the best candidate per example for the loss computation.
4. Perform the standard backward pass and parameter update.

At inference time, the model is used identically to a standard generative model—no additional sampling or ranking is needed. The gains from exploration are baked into the model parameters at training time.

## What the Guarantee Says

The authors provide a theoretical analysis showing that $\mathcal{L}_{\text{XM}}$ is an upper bound on the negative log-likelihood of the best-of-$K$ sample from the model. As $K \to \infty$, the objective converges to the entropy of the data distribution under the model's best mode, i.e., the model is incentivized to find and commit to the highest-probability mode rather than covering all modes. For $K = 1$, $\mathcal{L}_{\text{XM}}$ reduces to $\mathcal{L}_{\text{MLE}}$. Increasing $K$ strictly tightens the approximation of mode-committing behavior, providing a monotone scaling relationship.

## Experimental Findings

**Image generation (ImageNet 256×256, FID):** XM with $K=8$ achieves FID 1.43 without classifier-free guidance, versus 2.21 for the corresponding standard MLE baseline. With guidance, the XM baseline reaches 1.41. This is competitive with SOTA diffusion models trained with far more compute.

**Parameter efficiency:** XM reaches the FID of the standard model using only 53% of the parameters—a 47% parameter efficiency gain at matched training budget.

**Sample efficiency:** XM reaches the same FID as the standard model after only $1/6.2$ of the training samples.

**FLOP efficiency:** XM requires only $1/4.1$ of the training FLOPs to match the standard baseline's FID.

**Scaling law:** Increasing $K \in \{1, 2, 4, 8, 16, 32\}$ monotonically improves FID at matched parameter count and training compute. The gains are approximately log-linear in $K$, suggesting that exploration can be scaled similarly to parameters and data.

**Video generation:** On a text-to-video benchmark, XM reduces FVD by 21% relative to the MLE baseline at matched compute.

**Language generation:** On a causal language model fine-tuning task, XM improves diversity-quality Pareto frontier compared to standard fine-tuning, with 15% better MAUVE score.

## Ablations and Interpretation

**Similarity metric sensitivity:** Replacing perceptual loss with L2 distance for image tasks degrades performance, confirming that the quality of the match function matters. Perceptual metrics align better with human-perceived mode structure.

**$K$ vs.\ compute trade-off:** For a fixed compute budget, it is better to use $K=4$ with more training steps than $K=1$ with fewer steps and the same total FLOP count—exploration is more efficient than additional data processing.

**End-to-end generation:** The authors show XM supports training fully end-to-end generation models (e.g., encoder-decoder with shared parameters) that jointly produce intermediate representations and final outputs, because mode-committing training stabilizes the intermediate latent space.

**Connection to best-of-N:** Unlike best-of-N at inference (which requires $N$ full forward passes per generation), XM amortizes exploration into training. At inference, the model already has mode-committing behavior built in from training, requiring only a single forward pass.
