# StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference

**arXiv:** 2611.08363  
**Authors:** Jake Hyun, Ziyu Zhao, Jae-sun Seo, Mohamed S. Abdelfattah  
**Venue:** cs.LG / cs.DC  
**Submitted:** 5 November 2026  
**URL:** https://arxiv.org/abs/2611.08363

---

## Summary

Quantizing LLM weights and KV caches to 2–4 bits dramatically reduces peak GPU memory, enabling deployment of large models on constrained hardware, but introduces accuracy degradation that becomes severe at 2-bit precision. Prior work addresses this by increasing precision uniformly (e.g., GPTQ, AWQ) or by mixed-precision schemes that identify "outlier" weights and keep them in higher precision on the GPU. StrataFlux takes a different architectural view: the GPU's memory hierarchy — HBM, host DRAM, and NVMe SSD — provides multiple precision levels without increasing the peak GPU footprint. The method assigns strata of weights and KV cache entries to the appropriate memory tier based on their sensitivity, streams higher-precision copies of sensitive blocks during forward passes, and achieves near-full-precision accuracy at 2–3-bit average precision.

---

## Method

**Sensitivity profiling.** A one-shot calibration pass estimates per-layer weight sensitivity using a Fisher-information proxy. Layers with high sensitivity are assigned to a higher-precision stratum; the majority of layers remain at the target low bit-width.

**Memory-tier assignment.** Three strata:
- *Stratum A (GPU HBM):* The base 2-bit quantized weights, always resident.
- *Stratum B (host DRAM):* 4-bit copies of the top-k% most sensitive layers, paged on demand.
- *Stratum C (NVMe SSD):* 8-bit copies of a small subset of extremely sensitive layers (e.g., the first and last transformer blocks), paged asynchronously.

**Streaming pipeline.** A prefetch scheduler overlaps DMA transfers from DRAM/SSD with GPU computation on the previous layer, hiding the latency of tier-B and tier-C accesses behind normal forward-pass computation.

**KV-cache stratification.** The same tiering applies to KV cache entries. Recent tokens (small context window) stay in GPU HBM; older tokens are offloaded to DRAM and streamed back when attention needs them. This enables 128K+ context at reduced GPU memory.

---

## Experimental Setup

Models: Llama-3.1-8B-Instruct, Qwen3-8B.  
Baselines: standard 2-bit RTN, GPTQ, AWQ (4-bit), vLLM 4-bit AWQ with 8-bit KV.  
Benchmarks: WikiText-2 perplexity, AIME 2024/2025, long-context perplexity at 128K.  
Hardware: RTX 5090.

---

## Results

| Config | Model | Metric | Baseline | StrataFlux | Throughput |
|---|---|---|---|---|---|
| 2-bit weights | Llama-3.1-8B | WikiText-2 PPL | 9.46 | 8.40 | ~95% of base |
| 2-bit weights + KV | Qwen3-8B | AIME acc. | 11.0% | 47.1% | unchanged GPU mem. |
| 3-bit weights, 2-bit KV | Qwen3-8B (128K) | PPL | matches AWQ 4b/8b KV | matches | ½ GPU memory |

---

## Key Insight

The core insight is that memory-hierarchy bandwidth — not GPU compute — is the scarce resource during low-bit LLM inference, and that DRAM and SSD bandwidth, while slower than HBM, is sufficient to cover the small fraction of sensitive weights that need higher precision. By matching precision tiers to memory tiers rather than to GPU arithmetic units, StrataFlux avoids the accuracy-throughput trade-off that plagues purely GPU-resident mixed-precision schemes.

---

## Limitations

- Sensitivity profiling requires a calibration corpus; distribution mismatch between calibration and deployment can degrade precision tier assignments.
- The streaming pipeline's effectiveness depends on PCIe bandwidth; on systems with slow host-GPU interconnects the latency hiding may be insufficient.
- The approach is evaluated on a single high-end GPU (RTX 5090); results on multi-GPU or low-end deployments are not reported.
