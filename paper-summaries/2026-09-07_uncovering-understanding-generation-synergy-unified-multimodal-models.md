# Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models: From Representation, Task to System

**arXiv:** 2609.01607  
**Submitted:** September 1, 2026  
**Authors:** Penghao Wu, Haiwen Diao, Weichen Fan, Lewei Lu, Dahua Lin, Ziwei Liu  
**Affiliation:** S-Lab, Nanyang Technological University; SenseTime Research

---

## Headline Finding

Unified multimodal models benefit from joint visual understanding and generation training — but only when the two objectives are architecturally decoupled: forcing both through the same computation path causes one task to dominate and degrades overall performance.

---

## Key Findings (Pyramid Layer 2)

1. **Mutual benefit at the representation level.** Generation training enriches the visual features learned for understanding; understanding training strengthens vision-language alignment for generation. The synergy is real.
2. **Shared computation creates asymmetric degradation.** When both objectives share the same computation path, the stronger or more data-rich task suppresses the weaker one, negating the mutual benefit.
3. **Task-decoupled architecture resolves the conflict.** Specializing conflicting visual computation components for each task, while preserving shared semantic interaction layers, avoids dominance and yields consistent gains on both understanding and generation benchmarks.
4. **Three-level investigation framework.** The analysis proceeds from representation (feature-level mutual information), to task (benchmark accuracy), to system (end-to-end architecture design), providing a unified explanation for prior conflicting results in the unified multimodal model literature.

---

## Methodology (Pyramid Layer 3)

### Native Unified Multimodal Model Setup
The study controls for pretrained vision priors by training from scratch in a structurally native setting — both understanding and generation share a transformer backbone with no pretrained visual encoder. This isolates the effect of joint training from confounds introduced by frozen vision encoders.

### Three-Level Analysis

**Level 1: Representation**  
The authors measure cross-task mutual information between the learned visual representations: does training on generation improve the representations useful for understanding, and vice versa? They use probing classifiers and representation similarity analysis (CKA) to quantify the benefit.

**Level 2: Task**  
They systematically ablate joint vs. separate training across standard benchmarks (visual QA, image captioning, text-to-image generation), measuring accuracy and FID/CLIP-score.

**Level 3: System**  
They identify which architectural components cause the dominance conflict and propose a task-decoupled design that specializes those components while keeping others shared.

### Task-Decoupled Architecture
The key architectural insight is that conflict arises in visual encoding components (tokenizer, early attention layers) but not in semantic interaction layers (cross-attention, late transformer blocks). The proposed architecture:
- **Specialized visual encoders:** separate but lightweight branches for understanding vs. generation visual tokens.
- **Shared semantic backbone:** a single transformer stack that processes fused semantic representations from both branches.

---

## Technical Formulation

### Joint Training Objective
The standard joint objective is:
```
L_joint = λ_u · L_understanding + λ_g · L_generation
```
where:
- L_understanding is a cross-entropy loss over VQA / captioning tokens.
- L_generation is a diffusion denoising loss or autoregressive token prediction loss.
- λ_u, λ_g are loss weights.

### Dominance Metric
To quantify asymmetric degradation, the authors define a dominance score D:
```
D = (Acc_u_joint / Acc_u_separate) / (Acc_g_joint / Acc_g_separate)
```
where Acc_u and Acc_g are understanding and generation accuracies under joint vs. separate training. D >> 1 indicates understanding dominates; D << 1 indicates generation dominates.

### Decoupled Loss
In the task-decoupled architecture, visual representations are computed separately:
```
z_u = Encoder_u(x), z_g = Encoder_g(x)
z_shared = TransformerShared(z_u, z_g, text_tokens)
L_decoupled = L_u(z_shared) + L_g(z_shared)
```
The shared transformer sees information from both encoders but the encoders specialize independently.

---

## Experiments

### Understanding Benchmarks (VQA, image classification)
The task-decoupled model achieves an average 3.2% relative improvement over the best single-task baseline and a 5.8% improvement over the naive joint model.

### Generation Benchmarks (text-to-image FID on COCO)
The task-decoupled model achieves FID 8.1, compared to 10.3 for the joint baseline — a 21% relative reduction, approaching the performance of generation-specialized models.

### Parameter Overhead
The specialized visual encoders add ~12% extra parameters versus the fully shared model, but outperform it on both tasks simultaneously.

---

## Ablations and Interpretation

- **Specializing only early vs. late layers:** Specializing early tokenizer layers provides the largest gain; specializing late layers hurts by breaking shared semantic grounding.
- **Loss weight sensitivity (λ_u, λ_g):** The naive joint model is sensitive to loss weighting; the decoupled model is robust across a wide range of λ values.
- **Representation similarity (CKA):** After decoupled training, the understanding and generation visual representations become more dissimilar (lower CKA), but their projections into the shared semantic space converge — confirming the specialization hypothesis.
- **Scaling:** Gains persist at 300M, 1B, and 3B parameter scales, suggesting the architecture principle is scale-invariant.

---

## Reference

Penghao Wu, Haiwen Diao, Weichen Fan, Lewei Lu, Dahua Lin, and Ziwei Liu. **Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models: From Representation, Task to System.** arXiv:2609.01607, September 2026. https://arxiv.org/abs/2609.01607
