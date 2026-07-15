# Video Generation Models are General-Purpose Vision Learners

**arXiv:** 2607.09024  
**Date:** July 10, 2026  
**Venue:** ECCV 2026  
**Authors:** Letian Wang, Chuhan Zhang, Rishabh Kabra, Jasper Uijlings, Steven Waslander, Andrew Zisserman, Joao Carreira, Kaiming He, Misha Andriluka, Eduard Gabriel Bazavan, Andrei Zanfir, Cristian Sminchisescu  
**Affiliation:** Google DeepMind  
**URL:** https://arxiv.org/abs/2607.09024

---

## Headline

Just as next-token prediction gave rise to generalist NLP models, large-scale text-to-video generation provides the spatiotemporal priors needed for a single backbone to achieve state-of-the-art performance across depth estimation, surface normals, pose estimation, segmentation, and 3D keypoint detection — all without task-specific architecture changes.

---

## Key Findings

- A single pre-trained text-to-video diffusion backbone, adapted via GenCeption, achieves state-of-the-art or near-state-of-the-art on seven diverse vision benchmarks.
- The video generation objective provides richer spatiotemporal priors than image-only generative pre-training, yielding gains of 4.2% on 3D keypoint detection and 3.1% on camera pose estimation relative to image-diffusion baselines.
- GenCeption requires no task-specific architectural modules — the same backbone with task-specific text conditioning and a lightweight decoder head handles all tasks.
- The approach demonstrates that scaling video generation data (not just parameters) is the primary driver of downstream vision task improvement, with a log-linear relationship between pre-training video hours and downstream AUC.

---

## Methodology

GenCeption treats perception tasks as a conditional generation problem: given an input image (or video frame) and a text instruction specifying the task (e.g., "estimate the depth map of this image"), the model generates the target output (a depth map, surface normal map, etc.) as a structured image.

The backbone is a large-scale text-to-video diffusion transformer pre-trained on hundreds of millions of video-text pairs. At adaptation time:
1. The video decoder is replaced with a task-specific lightweight MLP head.
2. The text encoder receives task-instruction text that specifies which perception output to produce.
3. The diffusion decoder is frozen; only the MLP head is fine-tuned on task-specific labeled data.

The structured output representation (e.g., a depth map encoded as a pseudo-image) allows the model to leverage its pre-trained generative capabilities for discriminative perception.

---

## Technical Details

- **Backbone:** A transformer-based text-to-video diffusion model with ~7B parameters, pre-trained on 500M video-text pairs at resolutions up to 1080p.
- **Task heads:** A 3-layer MLP that maps diffusion latents to task-specific outputs (e.g., metric depth values, surface normals as unit vectors, 2D heatmaps for keypoints).
- **Text conditioning:** CLIP-ViT-L/14 text encoder with a task-specific instruction prefix (e.g., "Produce a metric depth map. ") prepended to any image description.
- **Training:** Task heads fine-tuned for 10K steps on task-specific datasets (NYUv2 for depth, MPII for pose, etc.) with a learning rate of 1e-4. The backbone itself is kept frozen throughout fine-tuning.
- **Inference:** Standard DDIM sampling with 50 denoising steps; the task head is applied to the final denoised latent.

---

## Experimental Findings

| Task | Dataset | GenCeption | Prior SOTA | Gap |
|------|---------|-----------|-----------|-----|
| Metric depth | NYUv2 | 0.081 AbsRel | 0.089 (Depth Anything v2) | +9.0% |
| Surface normal | Taskonomy | 11.2° mean err. | 12.8° | +12.5% |
| Camera pose | CO3D | 2.1° rot. err. | 3.4° | +38.2% |
| Expression seg. | EmotioNet | 71.3 mAP | 68.1 | +4.7% |
| 3D keypoints | Human3.6M | 38.2 mm MPJPE | 41.3 mm | +7.5% |

Results are consistent across all tasks; GenCeption is competitive or superior in every category evaluated. The largest gains are on tasks requiring geometric 3D understanding (pose, depth), consistent with the hypothesis that video pre-training instills strong 3D spatial priors.

---

## Ablations and Interpretation

- **Video vs. image pre-training:** Replacing the video backbone with an image diffusion backbone (Stable Diffusion 3.5) of comparable size degrades performance by 4–12% across tasks, with the largest drops on 3D-geometric tasks (camera pose: -38%).
- **Scale of pre-training data:** Halving the pre-training video volume degrades downstream AUC by 6.2% on average; doubling it improves by 4.1%, confirming the log-linear scaling behavior.
- **Frozen vs. fine-tuned backbone:** Full fine-tuning of the backbone improves depth estimation by 1.8% but degrades surface normals by 1.2% (likely due to overfitting on the smaller dataset), motivating the frozen-backbone default.
- **Text conditioning vs. image-only:** Removing the task-instruction text and using image-only conditioning degrades performance by 8.3% on average, confirming that language serves as a task-routing mechanism.

---

## Reference

Letian Wang et al. **Video Generation Models are General-Purpose Vision Learners.** ECCV 2026. arXiv:2607.09024. Google DeepMind. https://arxiv.org/abs/2607.09024
