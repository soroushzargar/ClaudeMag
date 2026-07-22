# Gemma 4 Technical Report

**arXiv:** 2607.02770
**Date:** July 2026
**Authors:** Gemma Team (Sherif El Abd et al.)
**Affiliation:** Google DeepMind
**URL:** https://arxiv.org/abs/2607.02770

---

## Headline

Google DeepMind releases Gemma 4, a new generation of fully open-weight natively multimodal language models (2.3B to 31B parameters) that introduces an encoder-free 12B architecture ingesting raw audio and image patches, a built-in thinking mode for explicit reasoning traces, and substantial advances on STEM, multimodal, and long-context benchmarks.

---

## Key Findings

- Gemma 4 establishes a new performance frontier for open-weight models on STEM, multimodal, and long-context benchmarks, matching or exceeding much larger frontier open models in human-rated evaluations.
- The 12B model introduces a unified encoder-free architecture that ingests raw audio and image patches end-to-end, eliminating the separate encoder modules of prior multimodal models.
- All model sizes include improved vision and audio encoders when operating in encoder-based mode, enabling stronger audio-visual understanding than prior Gemma versions.
- A built-in thinking mode enables all Gemma 4 models to generate explicit chain-of-thought reasoning traces before producing final answers, improving performance on multi-step tasks.
- Critical architectural improvements reduce inference memory, latency, and compute requirements, making the models practical to deploy on consumer and edge hardware.

---

## Methodology

Gemma 4 is trained across a family of architectures:

1. **Dense models (2.3B, 9B):** Standard dense Transformer architectures with improved depth-to-width ratios optimized for throughput and memory efficiency at small scales.
2. **MoE model (26B-A4B):** A Mixture-of-Experts model with 26B total parameters and 4B active parameters per token, balancing capability with inference cost.
3. **Encoder-free 12B model:** A novel unified architecture that directly processes raw image patches and audio frames through a shared token embedding space, without a dedicated visual or audio encoder.
4. **Thinking mode:** All models are post-trained to optionally prefix responses with extended reasoning traces using a reasoning budget mechanism that controls trace length.

---

## Technical Details

- **Encoder-free multimodality:** Image and audio inputs are tokenized using a learned patchification layer that maps patches to the same embedding dimension as text tokens, allowing a single attention mechanism to process all modalities.
- **MoE routing:** Gemma 4's 26B model uses top-$k$ expert routing with $k=2$ and 64 experts total. The routing function is $g(x) = \text{softmax}(x W_g)$, with load-balancing auxiliary loss $\mathcal{L}_{\text{aux}} = \alpha \sum_i f_i \cdot P_i$ where $f_i$ is expert load fraction and $P_i$ is average routing probability.
- **Long-context:** All models support context lengths up to 256K tokens through sliding-window attention interleaved with global attention layers.
- **Thinking mode training:** Supervised fine-tuning on synthetically generated reasoning traces, followed by RL with correctness reward applied to final-answer tokens only (not trace tokens), preventing reward hacking through trace inflation.
- **Quantization:** All models are released with QAT (Quantization-Aware Training) variants at 4-bit, enabling deployment on devices with as little as 8GB VRAM.

---

## Experimental Findings

- **MMLU-Pro (STEM):** Gemma 4 12B with thinking achieves 72.1%, competitive with models 3-5x larger.
- **MATH-500:** 91.4% with thinking mode enabled (Gemma 4 12B), up from 78.3% without thinking.
- **Multimodal benchmarks:** Strong performance on MMBench, SeedBench, and AI2D, with improvements of 4–9 points over Gemma 3 across all tested multimodal tasks.
- **Long-context (RULER):** Gemma 4 achieves near-perfect recall at 128K tokens and 87% at 256K tokens.
- **Audio understanding:** The encoder-free 12B model matches specialized audio models on speech recognition and audio captioning benchmarks, demonstrating that raw patch processing is competitive with dedicated encoder pipelines.
- **Human-rated evaluation:** Gemma 4 9B and 12B are preferred over models with 2–3x more parameters in head-to-head blind evaluations on open-ended generation tasks.

---

## Ablations and Interpretation

- **Encoder-free vs. encoder-based:** On standard image benchmarks, the encoder-free 12B model matches the encoder-based 12B model within 1–2 points, while reducing total model size and eliminating the encoder as a maintenance burden.
- **Thinking mode cost-benefit:** Enabling thinking mode on average doubles response latency but improves accuracy by 5–15 points on multi-step reasoning tasks; it is designed as an opt-in feature that can be disabled for latency-sensitive applications.
- **MoE vs. dense at comparable active compute:** The 26B-A4B MoE outperforms a 4B dense model on all benchmarks, confirming that total parameter count matters even when active compute is matched.
- **QAT quality retention:** 4-bit QAT variants retain 97–99% of the full-precision model's benchmark scores, enabling near-lossless quantization for on-device deployment.

---

## Reference

Gemma Team (Sherif El Abd et al.). **Gemma 4 Technical Report.** arXiv:2607.02770, July 2026. https://arxiv.org/abs/2607.02770
