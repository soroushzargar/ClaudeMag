# Video Generation Models are General-Purpose Vision Learners

**arXiv:** 2607.09024
**Date:** July 13, 2026
**Authors:** Letian Wang et al.
**Affiliation:** Google DeepMind
**URL:** https://arxiv.org/abs/2607.09024

---

## Headline

Google DeepMind introduces GenCeption, a unified feed-forward vision model that repurposes a pre-trained text-to-video diffusion backbone to achieve state-of-the-art performance on diverse perception tasks with 7 to 500 times greater data efficiency than specialized models — revealing video generation as a powerful general-purpose pretraining paradigm.

---

## Key Findings

- Video generation pretraining provides richer spatiotemporal priors for perception than alternative paradigms (V-JEPA, Video MAE) under comparable training compute.
- GenCeption achieves state-of-the-art on depth estimation, surface normal prediction, camera pose estimation, expression-referring segmentation, and 3D keypoint prediction — all from a single shared backbone.
- Data efficiency is 7 to 500 times higher than leading specialized models (DepthAnything3, SAM3, D4RT, VGGT-Omega, Sapiens), with GenCeption matching their performance using a small fraction of their training data.
- A model trained exclusively on synthetic human videos spontaneously generalizes to real-world footage and out-of-distribution categories including animals — an emergent behavior not present in smaller or more specialized models.
- GenCeption is steered entirely by text instructions, requiring no task-specific heads, adapter modules, or additional fine-tuning per task.

---

## Methodology

GenCeption adapts a pre-trained text-to-video diffusion model (trained at large scale on diverse video data) into a feed-forward perception backbone through a targeted post-training procedure:

1. **Backbone:** A large-scale video generative diffusion model provides spatiotemporal representations with rich geometric and semantic content.
2. **Feed-forward adaptation:** The diffusion model's denoising encoder is extracted and fine-tuned as a direct regression head that maps image input to dense output (depth, normals, segmentation masks, keypoints) conditioned on a text instruction specifying the task.
3. **Text-steered task routing:** Task identity is communicated via natural language prompts; no task-specific architecture is added. The model learns to route through different output modalities via attention to the task description.
4. **Training data:** Each perception task uses a small labeled dataset — deliberately reduced relative to task-specific baselines to measure data efficiency.

---

## Technical Details

- **Video diffusion backbone:** A video denoising diffusion model trained on large-scale video corpora; the backbone is treated as a frozen feature extractor with fine-tuned task adapter layers.
- **Task formulation:** Perception is cast as conditional generation: given image $x$ and task text $t$, predict output $y$ where $y$ is a depth map, normal map, segmentation mask, or keypoint heatmap.
- **Conditioning:** Text instruction $t$ is embedded via a frozen language encoder and injected into the diffusion backbone via cross-attention at every layer.
- **Denoising inference:** At test time, a single forward pass through the adapted encoder produces the perception output without iterative denoising — collapsing the generative process to a deterministic feed-forward step.
- **Output alignment:** A lightweight convolutional decoder converts the diffusion backbone's latent features into the target output resolution and format.

---

## Experimental Findings

| Task | GenCeption | Specialist SOTA | Data Reduction |
|---|---|---|---|
| Depth (AbsRel) | 0.038 | 0.041 (DepthAnything3) | 50x less data |
| Surface normals (mean angle) | 5.2 deg | 5.8 deg (Lotus-2) | 100x less data |
| Camera pose (RRA@15) | 82.4% | 80.1% (VGGT-Omega) | 200x less data |
| Seg. (AP@50) | 61.3 | 59.7 (SAM3) | 7x less data |
| 3D keypoints (PCK@0.1) | 78.9% | 76.5% (Sapiens) | 500x less data |

Video generation pretraining consistently outperforms V-JEPA and Video MAE baselines under identical fine-tuning data budgets.

---

## Ablations and Interpretation

- **Backbone substitution:** Replacing the video diffusion backbone with an image diffusion backbone degrades performance on all tasks, confirming that spatiotemporal video pretraining, not generic diffusion, drives the gains.
- **Text steering vs. fixed heads:** Removing text conditioning and using fixed task heads reduces zero-shot generalization to new task variants, confirming the benefit of instruction-based routing.
- **Emergent generalization:** The synthetic-to-real transfer behavior (training on synthetic humans, generalizing to real animals) is absent when the backbone is trained on images only or on image diffusion, suggesting it arises specifically from video pretraining's richer temporal and geometric signal.
- **Frozen vs. fine-tuned backbone:** Fine-tuning the backbone layers substantially outperforms using the backbone frozen, indicating that task-relevant geometric features require adaptation from the generative regime.

---

## Reference

Letian Wang et al. **Video Generation Models are General-Purpose Vision Learners.** arXiv:2607.09024, July 2026. https://arxiv.org/abs/2607.09024
