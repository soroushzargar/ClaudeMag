# DiffusionGemma Technical Report

**arXiv:** 2608.00146  
**Authors:** DiffusionGemma Team (Adrien Ali Taïga et al., 43 authors), Google DeepMind  
**Date:** August 2026  
**URL:** https://arxiv.org/abs/2608.00146

---

## Summary

DiffusionGemma is Google DeepMind's experimental open-weight language model that generates text using discrete diffusion rather than conventional left-to-right autoregressive decoding. Instead of emitting one token per step, it iteratively refines an entire block of 256 tokens in parallel—a strategy called block-autoregressive multi-canvas sampling—eliminating the sequential decoding bottleneck that limits all AR models to generating one token at a time.

## Architecture

The model is built on the Gemma 4 Mixture-of-Experts backbone with 26B total parameters and only 3.8B activated at inference time (the 26B-A4B variant). The sparse MoE design means most parameters are dormant during any given forward pass, keeping latency low while maintaining capacity. Because the model processes both text and image/video inputs to produce text outputs, it is natively multimodal.

## Training Pipeline

Fine-tuning from the pre-trained Gemma 4 AR checkpoint uses less than 10% of the original pre-training token budget and proceeds in two stages:

1. **Supervised Fine-Tuning (SFT):** Teaches the model bidirectional denoising—unlike an AR model, DiffusionGemma must learn to fill in masked tokens conditioned on both left and right context within the canvas, a fundamentally different skill from left-to-right generation.

2. **RL + Sampler Distillation:** A combined reinforcement learning and distillation stage jointly improves generation quality (by rewarding outputs preferred by a judge) and inference efficiency (by training the sampler to converge in fewer denoising steps).

## How Discrete Diffusion Works

At inference time DiffusionGemma begins with a fully masked canvas of 256 tokens. Each denoising iteration predicts the probability distribution over all masked positions simultaneously, samples values for the most confident positions, and repeats. Positions committed in earlier steps are never revised. After a small number of iterations (typically far fewer than 256) the canvas is fully populated. Because tokens are revealed in parallel rather than serially, throughput scales with hardware parallelism rather than sequence length.

## Performance

- **1000+ tokens/second** on a single NVIDIA H100
- **700+ tokens/second** on NVIDIA GeForce RTX 5090
- **Up to 4× faster** than comparable autoregressive models on dedicated GPUs
- Fits within **18 GB VRAM** when quantized to 4-bit, enabling local deployment on consumer GPUs

The model is released under the **Apache 2.0 license**.

## Why It Matters

DiffusionGemma represents a significant step toward making diffusion-based text generation practical. Autoregressive decoding has been the default paradigm for LLMs precisely because earlier diffusion text models were too slow or low quality. By starting from a strong AR pre-trained foundation and adapting it with a compute-frugal fine-tuning recipe, the DiffusionGemma approach shows that the gap can be closed without the enormous cost of training a diffusion LM from scratch. The 4× throughput improvement at matched quality changes the cost structure of inference at scale.

## Key Contributions

1. Block-autoregressive multi-canvas sampling applied to a strong MoE foundation model
2. Two-stage fine-tuning recipe requiring <10% of original training compute
3. RL-guided sampler distillation for simultaneous quality and efficiency improvement
4. Open-weight release enabling community research into discrete diffusion LMs
