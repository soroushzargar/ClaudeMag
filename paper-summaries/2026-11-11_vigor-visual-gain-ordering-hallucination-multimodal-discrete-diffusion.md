# VIGOR: Visual Gain Ordering for Hallucination Mitigation in Multimodal Discrete Diffusion Language Models

**arXiv:** 2611.05066  
**Authors:** NeurIPS 2026 Submission (anonymous)  
**Venue:** NeurIPS 2026  
**Submitted:** 3 November 2026  
**URL:** https://arxiv.org/abs/2611.05066

---

## Summary

Multimodal discrete diffusion language models (dLLMs) such as LLaDA-V and Lumina-DiMOO generate text by iteratively unmasking token positions, choosing at each step which positions to reveal based on model confidence. The authors identify a root cause of hallucination in these models: the unmasking confidence for many positions is dominated by language priors rather than visual evidence. A position that fits naturally in the sentence given surrounding text will be unmasked early regardless of what the image shows, causing the model to commit to language-prior predictions before visual evidence can influence later positions. VIGOR is a training-free, plug-and-play inference strategy that corrects this by reordering unmasking according to counterfactual visual gain — prioritizing positions whose confidence is most strongly supported by the image.

---

## Method

**Visual gain score.** At each denoising step, VIGOR computes two forward passes through the model:
1. *Image-conditioned pass:* The normal pass with image I and text context T, yielding confidence scores P(token | I, T) for each masked position.
2. *Visual-ablation pass:* A pass with the image replaced by a null embedding (or masked image tokens), yielding P(token | ∅, T).

The visual gain for position j is defined as:
  VG(j) = P(ŷ_j | I, T) - P(ŷ_j | ∅, T)

where ŷ_j = argmax P(token | I, T). A high visual gain indicates that the model's prediction for position j is specifically supported by the image content. A low or negative visual gain indicates the prediction comes primarily from language priors.

**Reordering.** Rather than unmasking in order of raw confidence (the standard strategy in masked dLLMs), VIGOR unmasks in order of visual gain. The top-k positions by visual gain are unmasked at each step. This ensures that image-grounded tokens are committed to first, providing an anchor that influences subsequent token predictions.

**Plug-and-play deployment.** VIGOR requires no model fine-tuning. It adds two forward passes per denoising step (one ablation pass in addition to the standard pass), and the reordering adds negligible compute overhead beyond the extra pass. The null image embedding can be a learned or fixed zero vector, and results are robust to this choice.

---

## Experimental Setup

Evaluated on:
- **Backbones:** LLaDA-V (masked dLLM-based multimodal model) and Lumina-DiMOO
- **Benchmarks:** Standard VQA, multimodal reasoning, and hallucination-specific benchmarks (e.g., POPE, MMHal-Bench, NocCaps)
- **Baselines:** Standard unmasking (confidence-ordered), other training-free hallucination mitigation methods for autoregressive VLMs adapted to the dLLM setting

---

## Results

VIGOR consistently reduces hallucination rates and improves multimodal reasoning performance across all evaluated benchmarks and both backbone models. Gains are largest on benchmarks that specifically measure object-level hallucination (POPE-style evaluation), where grounding visual tokens early is most consequential. On general VQA benchmarks, VIGOR improves or matches baseline performance, confirming that the reordering does not hurt performance on tasks without hallucination risk.

---

## Key Insight

The fundamental observation is that discrete diffusion models are not just autoregressive models in disguise — their iterative unmasking creates a commitment order that influences which tokens get the benefit of which contextual cues. By placing visually-grounded tokens first in the commitment order, VIGOR ensures that the model's visual predictions set the stage for subsequent language-driven predictions, rather than the reverse. This insight connects the hallucination problem to the sequential structure of the denoising process in a way that makes training-free intervention possible.

---

## Limitations

- The ablation pass doubles inference cost per step; in latency-sensitive applications this overhead may be prohibitive.
- The null image embedding choice (zero vector vs. learned null token) may affect performance on models with different image tokenization schemes.
- VIGOR addresses the ordering of unmasking but not the quality of the model's visual representations; hallucinations arising from inadequate visual encoding are not targeted.
