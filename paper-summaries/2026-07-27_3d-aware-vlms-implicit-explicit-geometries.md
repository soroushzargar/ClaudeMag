# 3D-Aware VLMs with Implicit and Explicit Geometries

**Authors:** Wenhao Li, Xueying Jiang, Quanhao Qian, Deli Zhao, Ran Xu, Shijian Lu, Gongjie Zhang  
**Venue:** ECCV 2026 / arXiv:2607.21595  
**Date:** July 23, 2026  
**URL:** https://arxiv.org/abs/2607.21595

---

## Summary

Most vision-language models (VLMs) are trained predominantly on 2D image-text data and struggle with tasks requiring fine-grained 3D spatial understanding. This paper introduces **VLM-IE3D**, a unified framework that equips VLMs with both *implicit* and *explicit* 3D geometric representations learned purely from RGB video—no depth sensors or 3D supervision required. The dual-geometry design achieves state-of-the-art results on 3D video detection, 3D visual grounding, 3D dense captioning, and spatial reasoning benchmarks, accepted at ECCV 2026.

## Problem

State-of-the-art VLMs such as LLaVA and InternVL acquire some 3D awareness through scale-free visual pretraining, but this results in only a coarse "cognitive map" of scene geometry. Existing 3D-aware VLM approaches either:
- Require explicit 3D sensor input (depth, LiDAR) at inference time, limiting deployment to sensor-equipped platforms.
- Rely solely on monocular depth estimation, which provides dense but noisy geometry without structural guarantees.
- Treat 3D awareness as an add-on via a single geometric modality, missing the complementary benefits of combining coarse global geometry with fine-grained local geometry.

## Method: VLM-IE3D

VLM-IE3D introduces two complementary geometric representations, each encoded as visual tokens fed to the VLM's transformer:

### Implicit Geometry Tokens (IGTs)
IGTs capture coarse, high-level geometric priors inferred from temporal video context. They are produced by a lightweight video geometry encoder that processes multiple frames and outputs a compact set of tokens encoding relative depth ordering, surface orientation statistics, and camera ego-motion. IGTs provide the model with a "3D cognitive map"—holistic scene layout without structural precision.

### Explicit Geometry Tokens (EGTs)
EGTs encode fine-grained, locally structured 3D geometry derived from monocular video reconstruction. A differentiable structure-from-motion (SfM) module recovers a sparse point cloud and surface normals from the input video frames. These are projected into a fixed-resolution 3D feature volume and then encoded into EGTs via a 3D sparse convolutional encoder. EGTs provide precise local geometry—object boundaries, surface textures, and depth discontinuities—complementing the coarse global layout of IGTs.

### 3D-Aware Adapter
A cross-attention adapter fuses IGTs and EGTs with the 2D visual tokens from the VLM's image encoder. The adapter uses geometry-conditioned attention masks that allow EGTs to attend selectively to spatially aligned 2D regions, preventing geometric noise from interfering with semantic understanding.

## Technical Formulation

Let $V = \{f_1, \ldots, f_T\}$ be an RGB video of $T$ frames. The 2D visual encoder produces visual tokens $Z_\text{2D} = \phi_{2D}(V) \in \mathbb{R}^{N_\text{2D} \times d}$.

The implicit geometry encoder produces:

$$Z_\text{IGT} = \phi_\text{IG}(V) \in \mathbb{R}^{N_\text{IGT} \times d}$$

The SfM module recovers a point cloud $\mathcal{P} = \{(\mathbf{p}_i, \mathbf{n}_i)\}$ where $\mathbf{p}_i \in \mathbb{R}^3$ and $\mathbf{n}_i$ is the surface normal. The explicit geometry encoder produces:

$$Z_\text{EGT} = \phi_\text{EG}(\mathcal{P}) \in \mathbb{R}^{N_\text{EGT} \times d}$$

The 3D-aware adapter computes fused tokens via:

$$Z_\text{fused} = \text{Attn}(Q = Z_\text{2D},\; K = V = [Z_\text{IGT}; Z_\text{EGT}; Z_\text{2D}]) + Z_\text{2D}$$

with geometry-conditioned masks ensuring that EGT tokens attend only to 2D tokens within a spatial radius $r$ in image space.

The full fused representation $Z_\text{fused}$ is passed to the frozen VLM language backbone along with the text query.

## Experimental Findings

VLM-IE3D is evaluated on four 3D benchmark families:

| Benchmark | Task | VLM-IE3D | Best Prior |
|---|---|---|---|
| ScanQA | 3D Spatial Reasoning | 52.7 | 46.3 |
| ScanRefer | 3D Visual Grounding (Acc@0.5) | 48.9 | 41.2 |
| Scan2Cap | 3D Dense Captioning (CIDEr@0.5) | 71.3 | 62.8 |
| EmbodiedScan (det) | 3D Detection (mAP@0.25) | 33.1 | 28.4 |

All gains are achieved with RGB-only input, matching or exceeding methods that use ground-truth depth at inference.

## Ablation Results

- Removing IGTs (EGT only) degrades ScanQA by 4.1 points, confirming that coarse global layout is necessary for spatial reasoning.
- Removing EGTs (IGT only) degrades ScanRefer grounding by 6.2 points, confirming that fine-grained local geometry is necessary for precise localization.
- Removing the 3D-aware adapter and concatenating tokens naively reduces performance by 3.8 points on average, demonstrating the value of geometry-conditioned fusion.

## Significance

VLM-IE3D shows that combining complementary geometric representations—global layout (IGTs) and local structure (EGTs)—substantially improves 3D spatial understanding in VLMs without requiring any 3D sensor hardware. The RGB-only design makes the approach deployable in standard camera settings.
