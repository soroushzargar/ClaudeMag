# Rectified Language Flow

**arXiv:** 2610.33811  
**Authors:** Baiyu Su, Runlong Liao, Lizhang Chen, Qiang Liu (UT Austin)  
**Submitted:** October 24, 2026  
**Venue:** arXiv preprint (cs.AI, cs.LG)  
**URL:** https://arxiv.org/abs/2610.33811

---

## Summary

Autoregressive (AR) language models generate tokens one at a time; each forward pass produces a single next-token distribution. This serial dependency is the fundamental bottleneck at inference time. Discrete diffusion language models decouple generation from left-to-right order but introduce a different cost: dozens of denoising steps over a discrete token space, each requiring a full model forward pass.

Rectified Language Flow (ReLF) proposes a third path: flow matching in a continuous embedding space. Rather than working in discrete token space or a variational latent, ReLF operates in the space of token embeddings and trains a velocity field that moves noise to embeddings along a straight (rectified) path. The rectified trajectory is key: it enables accurate generation with far fewer function evaluations than curved-path flows or diffusion models.

---

## Method

**Hellinger-SVD Embeddings.** The paper fixes token embeddings by applying SVD to the Hellinger-transformed co-occurrence matrix of the training corpus, producing a geometry in which the flow can be trained stably without embedding collapse.

**Self-Conditioning Guidance.** During training the model conditions on a preliminary self-estimate of the clean data point. This is trained alongside the flow with a cross-entropy objective, providing a noisy but useful "target hint" that helps the model resolve ambiguous intermediate flow states.

**Regularized Reflow.** Reflow (or rectified flow distillation) progressively straightens the learned trajectories so that fewer function evaluations are needed at test time. A regularization term prevents the reflow stage from creating degenerate shortcuts that hurt diversity.

**Singular Logits.** As the flow approaches t = 1 (the data manifold), the embeddings must be decoded to discrete tokens. ReLF introduces "singular logits," a differentiable mechanism that handles the final commitment to a discrete token while remaining compatible with gradient-based training and the continuous flow objective.

---

## Results

- On OpenWebText, ReLF achieves **lower generative perplexity than autoregressive baselines** at matched diversity scores, across three model sizes.
- It **outperforms prior continuous flow methods** (ELF and variants) across all evaluated model sizes.
- With 4 function evaluations, ReLF generates 1,024-token sequences **15–19× faster** than a matching AR model at batch size 1, with minimal perplexity degradation.
- With 64 function evaluations it achieves the best reported perplexity on the OpenWebText test set among non-AR models of the same parameter count.

---

## Significance

ReLF occupies a compelling niche: it keeps the expressivity of a sequence model while enabling large inference speedups. The Hellinger-SVD embedding is a clean, parameter-free initialization that sidesteps the instabilities of learned continuous embeddings reported in prior work. The singular-logits mechanism cleanly separates the continuous-flow regime from the final discrete commitment—a design that previous flow LM papers left implicit or handled with a separate projector. If the perplexity results hold under more rigorous comparison, ReLF would represent the best-performing fast language model to date.

---

## Limitations

- Evaluated only on OpenWebText; no long-context or instruction-following benchmarks.
- The reflow distillation stage adds training cost and its interactions with the regularizer are characterized only empirically.
- The 15–19× speedup is measured at batch size 1; gains at large batch sizes typical of production inference may differ.
