# CMuon: Accelerating and Stabilizing Diffusion Transformer Training via Chunked Momentum Orthogonalization

**arXiv:** 2608.02502  
**Authors:** Chuyan Chen, Peng Sun, Kun Yuan  
**Venue:** ECCV 2026  
**Date:** August 2026  
**URL:** https://arxiv.org/abs/2608.02502

---

## Summary

Diffusion Transformers (DiTs) have become the dominant architecture for high-quality image and video synthesis, yet their training demands enormous compute. The recently proposed Muon optimizer (Momentum Orthogonalization) offers faster convergence than AdamW for standard transformers, but its direct application to DiTs yields suboptimal late-stage convergence. CMuon (Chunked Muon) identifies the root cause—implicit subspace coupling induced by fused weight tensors—and fixes it with a single, targeted modification: partition fused matrices into independent chunks before applying orthogonalization. The result is a 675M-parameter DiT that achieves FID 1.18 on ImageNet 256×256 in 200 training epochs, more than 2× faster than AdamW.

## Background: Muon and Its Problem with DiTs

Muon updates weight matrices by orthogonalizing their momentum, ensuring each update step moves in a direction that maximally uses its available "degrees of freedom" in weight space. For standard Transformer layers (attention and MLP blocks), Muon works well. But DiTs employ Adaptive Layer Norm (AdaLN)—a conditioning mechanism that modulates activations via learned scale and shift parameters derived from a timestep embedding. In practice, the weight matrices for different functional components (e.g., the scale and shift within AdaLN, or the Q, K, V projections in attention) are often fused into single large tensors for GPU efficiency. Applying Muon to these fused tensors treats functionally independent subspaces as a single entity, causing the orthogonalization to mix their update directions in ways that harm optimization.

## The CMuon Fix

The fix is conceptually minimal: before computing the Nesterov momentum and orthogonalizing it, CMuon splits fused matrices along their channel dimension according to their functional roles. Each chunk is orthogonalized independently. The chunks are then reassembled before the weight update. This preserves the GPU efficiency of fused kernels while eliminating the subspace coupling that causes late-stage convergence failure in vanilla Muon.

No hyperparameter changes or architectural modifications are required beyond the chunking step.

## Results

Evaluated on the standard DiT-XL/2 architecture (675M parameters) trained on ImageNet 256×256:

- **FID 1.18** in 200 epochs (state-of-the-art at this scale with no guidance at inference time)
- **>2× training speedup** over AdamW measured by FID-vs-epoch
- **No late-stage convergence plateau** that plagued vanilla Muon applied to DiTs
- Stable training curves across the full 200-epoch run

## Why It Matters

DiT training is expensive enough that optimizer improvements translate directly to cost savings at scale. A 2× speedup means halving GPU-hours to reach a target FID, which at the scale of frontier image/video generation models represents millions of dollars of compute. The root-cause analysis (fused weights break Muon's assumption of independent subspaces) is also a broadly applicable insight: any architecture that fuses functionally distinct weights for efficiency may exhibit similar issues with second-order or orthogonalization-based optimizers.

## Key Contributions

1. Root-cause analysis linking DiT weight fusion to Muon's convergence degradation
2. CMuon: the chunked matrix partitioning fix that restores correct orthogonalization
3. FID 1.18 on ImageNet 256 at 675M parameters in 200 epochs (>2× AdamW speedup)
4. Accepted to ECCV 2026
