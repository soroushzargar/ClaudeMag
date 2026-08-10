# Qwen-3D: A Generalist 3D Vision-Language Model for Spatial Understanding

**arXiv:** 2608.02980
**Authors:** Lucy Lin, Ayush Jain, Yifan Liu, Katerina Fragkiadaki (Carnegie Mellon University)
**Venue:** ECCV 2026
**Date:** August 2026
**URL:** https://arxiv.org/abs/2608.02980
**Project:** https://qwen-3d.github.io/

---

## Summary

Qwen-3D is a generalist 3D vision-language model from Carnegie Mellon University that unifies language reasoning, 2D grounding, and 3D grounding in a single architecture. Its key innovation is performing attention directly in 3D world space rather than over independent image frames, achieving state-of-the-art results on a broad suite of vision-language and 3D understanding benchmarks while outperforming several large proprietary 2D models.

## Problem

Large Multimodal Models (LMMs) have achieved strong results on images and short videos, but scaling to long videos and 3D scenes remains difficult. Frame-centric tokenization treats each image or video frame independently, ignoring the shared 3D structure underlying all views. Existing 3D LMMs incorporate some geometry awareness but still lag behind specialist 3D perception systems on grounding and segmentation tasks, and cannot generalize well across both 2D and 3D modalities from a single model.

## Key Idea: 3D World-Space Attention

Given multi-view RGB observations, depth maps, and camera poses, Qwen-3D maps all visual tokens into a shared 3D coordinate system. Instead of treating each frame's tokens as a flat 2D sequence, the model applies geometry-aware attention through **3D Rotary Positional Embeddings (3D RoPE)**: each token's position encoding reflects its 3D world coordinate rather than its pixel location in a frame.

This allows observations from multiple views and time steps to be fused into a **persistent, world-aligned representation**. Two tokens that correspond to the same 3D point—even if captured by different cameras or at different times—receive positional encodings that reflect their proximity in world space, enabling the model to naturally associate them during attention.

## Architecture

- **Backbone:** Builds on Qwen-VL, a strong vision-language foundation
- **3D Tokenization:** Depth and camera pose lift 2D pixel locations to 3D world coordinates
- **3D RoPE:** Rotary positional embeddings indexed by (X, Y, Z) world position instead of (row, col) pixel position
- **Unified Output Head:** A single model jointly predicts language, 2D bounding boxes, and 3D object locations (center, size, orientation)

## Training

Qwen-3D is trained jointly on both 2D and 3D data using a multi-task objective:
- Language generation (instruction following, VQA)
- 2D grounding (bounding box prediction from text queries)
- 3D grounding (3D bounding box prediction from text queries)

Joint training on 2D and 3D data is critical: it prevents catastrophic forgetting of strong 2D capabilities while learning the novel 3D understanding tasks.

## Results

- **State-of-the-art on 3D understanding benchmarks** (ScanQA, EmbodiedScan, and others), surpassing prior 3D LMMs by significant margins
- **Competitive on 2D vision-language benchmarks** (MMBench, MMStar, SEED-Bench), maintaining strong 2D performance despite 3D specialization
- **Outperforms several large proprietary 2D models** on 3D grounding tasks despite being a generalist model
- **Surpasses specialist 3D perception systems** on grounding and segmentation benchmarks, closing the gap between generalist VLMs and task-specific perception pipelines

## Why It Matters

3D spatial understanding is fundamental for embodied AI, robotics, and augmented reality applications. Prior work required separate specialist models for 2D and 3D tasks. Qwen-3D shows that a single model can unify both modalities by simply changing where attention is computed—from 2D image space to 3D world space—without adding complexity to the architecture. The 3D RoPE mechanism is elegant and compatible with standard transformer training pipelines, making it likely to be widely adopted.

## Key Contributions

1. 3D Rotary Positional Embeddings that index attention by world-space coordinates
2. A world-aligned persistent representation fusing multi-view and multi-time observations
3. Unified architecture jointly supporting language, 2D grounding, and 3D grounding
4. ECCV 2026 result demonstrating generalist 3D VLMs can match specialist perception systems
