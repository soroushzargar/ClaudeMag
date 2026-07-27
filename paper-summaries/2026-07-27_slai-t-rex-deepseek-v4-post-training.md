# SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD

**Authors:** SLAI Team (Huawei Cloud AI Lab)  
**Venue:** arXiv:2607.20145  
**Date:** July 2026  
**URL:** https://arxiv.org/abs/2607.20145

---

## Summary

Training trillion-parameter Mixture-of-Experts (MoE) models has been the exclusive province of NVIDIA GPU clusters. This paper demonstrates that full-parameter post-training of the DeepSeek-V4 family—a 671-billion-parameter open-weight MoE model—can be performed efficiently on Huawei Ascend NPU SuperPOD clusters through a hierarchical optimization framework. The resulting system achieves 34.22% Model FLOPs Utilization (MFU), a 2.93× improvement over the open-source GPU baseline, and enables the development of an OR-specialist model that outperforms GPT-5.4-Mini by 3.98 percentage points on operations research benchmarks.

## Problem

Full-parameter post-training of trillion-scale MoE models on alternative hardware introduces three compounding challenges:

1. **Memory pressure:** Trillion-parameter models require distributing parameters, gradients, and optimizer states across hundreds of devices. MoE's expert routing introduces irregular memory access patterns.

2. **Communication overhead:** Expert parallelism requires all-to-all communications for token routing. On Ascend clusters, the communication topology differs from NVLink-based GPU clusters, making naive adaptation of GPU pipelines highly inefficient.

3. **Kernel inefficiency:** Ascend NPU's CANN compiler ecosystem and cube-tile memory hierarchy require NPU-specific kernel implementations—straightforward GPU-to-NPU translation leaves significant compute on the table.

## Method: SLAI T-Rex Optimization Framework

The SLAI T-Rex framework addresses these challenges through a three-level hierarchical design:

### Level 1: Model-Level Parallelism
The framework implements a custom expert-parallel strategy combining:
- **Expert Shard Parallelism (ESP):** Each expert's parameters are sharded across multiple NPU pods, reducing peak memory per device.
- **Pipeline Depth Reduction:** A novel pipeline schedule reduces bubble overhead by restructuring micro-batch ordering to overlap expert computation with inter-pod communication.
- **Selective Activation Checkpointing:** Only attention activations (not expert activations) are checkpointed, trading a small re-computation cost for major memory savings without sacrificing throughput.

### Level 2: Computation-Communication Orchestration
Expert routing in MoE requires all-to-all token dispatches between pods. SLAI T-Rex introduces a **communication-aware scheduler** that:
- Predicts expert load imbalance using lightweight routing statistics from recent steps.
- Adjusts micro-batch boundaries to reduce hot-expert serialization.
- Overlaps all-to-all dispatches with local attention computations via asynchronous execution streams.

### Level 3: Low-Level Kernel Optimization
Custom CANN kernels are implemented for:
- Expert gating with fused softmax and top-k selection.
- Flash attention adapted to Ascend's cube tile architecture.
- Gradient all-reduce with dynamic bucketing tuned for Ascend's high-bandwidth memory (HBM) bandwidth profile.

## Technical Formulation

For a MoE layer with $E$ experts, token routing is governed by:

$$g(x) = \text{TopK}\!\left(\text{softmax}(W_g x),\, k\right)$$

where $W_g \in \mathbb{R}^{E \times d}$ is the gating matrix and $k$ is the number of active experts per token. The MoE layer output is:

$$\text{MoE}(x) = \sum_{e \in \text{TopK}(x)} g_e(x) \cdot \text{FFN}_e(x)$$

Expert parallel training requires that each training step communicates $O(B \cdot k \cdot d)$ values across pods (where $B$ is batch size and $d$ is the token dimension). SLAI T-Rex reduces effective communication volume by fusing the dispatch and combine operations and prefetching expert weights from HBM.

MFU is defined as:

$$\text{MFU} = \frac{\text{Observed FLOPs/sec}}{\text{Peak Theoretical FLOPs/sec (NPU)}}$$

and is measured as 34.22%, compared to 11.68% for the open-source baseline on the same cluster.

## Application: OR-Specialist Model

Building on the optimized infrastructure, the paper demonstrates an end-to-end CPT+SFT workflow for operations research (OR) tasks:
- **Continued Pretraining (CPT):** The base DeepSeek-V4-Flash model is continued-pretrained on a curated corpus of OR literature, textbooks, and synthetic problem instances.
- **Supervised Fine-tuning (SFT):** 10,000 high-quality SFT samples spanning LP, MILP, scheduling, and routing problems with three problem representations (mathematical, natural language, structured JSON).

The resulting OR-specialist achieves 71.81% zero-shot pass@1, outperforming:
- GPT-5.4-Mini: 67.83% (+3.98 pp)
- DeepSeek-V4-Flash (base): 60.54% (+11.27 pp)

## Significance

SLAI T-Rex establishes that the Ascend NPU ecosystem is now a viable platform for state-of-the-art LLM post-training at trillion-parameter scale. The 2.93× MFU improvement bridges a substantial efficiency gap, and the OR-specialist results demonstrate that domain-specialized post-training on alternative hardware can produce models that compete with closed frontier models in narrow but commercially important domains.
