# Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs

**arXiv:** 2608.04048
**Authors:** Yu Luo, Bo Dong, Wenhua Cheng, Haihao Shen
**Venue:** NeurIPS 2026
**Date:** August 2026
**URL:** https://arxiv.org/abs/2608.04048

---

## Summary

Recurrent Residual Quantization (RRQ) is a post-training quantization framework that enables deploying a single LLM checkpoint at multiple precisions (2-, 4-, 6-, and 8-bit) without separate quantization runs for each bit-width. Starting from a 2-bit base, RRQ progressively adds lightweight 2-bit residual corrections, is calibration-free, and constructs the full multi-precision package 3.3× faster than existing methods. Accepted to NeurIPS 2026.

## Problem

Production deployments of LLMs face diverse hardware constraints: a data center GPU can run 8-bit inference at high throughput, while an edge device may only fit a 2-bit model. Today this requires maintaining and storing a separate quantized checkpoint for each target bit-width, with each checkpoint requiring a separate (often expensive and calibration-data-dependent) quantization run. This creates storage, maintenance, and latency overhead that scales linearly with the number of target precisions.

## Key Idea: Progressive Residual Corrections

RRQ represents LLM weights as a **quantized base plus a recurrent sequence of quantized residual corrections**:

$$w \approx Q_2(w) + Q_2(w - Q_2(w)) + Q_2\!\left(w - Q_2(w) - Q_2(w - Q_2(w))\right) + \ldots$$

Each correction term is obtained by quantizing the residual error of the previous approximation to 2 bits via round-to-nearest (RTN). Adding $k$ correction terms yields an effective $(2 + 2k)$-bit representation:
- Base only: effective 2-bit
- Base + 1 residual: effective 4-bit
- Base + 2 residuals: effective 6-bit
- Base + 3 residuals: effective 8-bit

All precisions share the same base; higher precisions simply read more correction terms from the same stored sequence. A single serialized checkpoint stores the base and all residuals; inference at a given precision reads only the required prefix.

## Technical Properties

- **Calibration-free:** The base and each residual correction are computed independently via RTN, requiring no calibration dataset. This avoids the compute and data dependencies of methods like GPTQ or AWQ.
- **No joint multi-bit optimization:** Each residual is computed greedily on the residual error of the previous step, keeping construction time low.
- **Storage-efficient:** The storage cost is proportional to the maximum target precision; storing the 8-bit package (base + 3 residuals) costs approximately the same as a single 8-bit quantized model.

## Construction Speed

For Qwen3-8B, the complete 2/4/6/8-bit RRQ package is constructed in **1,293 seconds**—**3.3× faster** than MatGPTQ, the state-of-the-art calibration-based quantization method for the same model. The speedup comes from eliminating the calibration data pass and the expensive second-order optimization that calibration-based methods require.

## Results

- **Multiple effective precisions from one checkpoint**, enabling hardware-adaptive deployment
- **3.3× faster construction** than MatGPTQ on Qwen3-8B
- Competitive perplexity with calibration-based baselines at 4-bit and above
- Minimal quality degradation at 6- and 8-bit compared to FP16 baseline

## Why It Matters

As LLMs are deployed across heterogeneous hardware—from cloud TPUs to laptop CPUs—the ability to serve a single model at multiple precisions from one checkpoint simplifies the entire deployment pipeline. RRQ's calibration-free approach removes a key dependency (calibration data) that can be expensive or unavailable in practice, and the progressive structure means operators can trade accuracy for speed at runtime without re-quantizing the model.

## Key Contributions

1. Progressive residual quantization scheme enabling multiple precisions from one checkpoint
2. Calibration-free construction via greedy RTN residuals
3. 3.3× construction speedup over GPTQ-class baselines
4. Theoretical analysis of the progressive approximation error
