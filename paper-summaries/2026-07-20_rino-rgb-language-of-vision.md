# Let RGB Be the Language of Vision (RINO)

**arXiv:** 2607.12450  
**Date:** July 14, 2026  
**Authors:** Timing Yang, Jinrui Yang, Xinlong Li, Yuhan Wang, Haoran Li, Yanqing Liu, Guoyizhe Wei, Jixuan Ying, Chen Wei, Rama Chellappa, Yuyin Zhou, Cihang Xie, Alan Yuille, Feng Wang  
**Affiliation:** Johns Hopkins University, UC Santa Cruz, Carnegie Mellon University, Rice University  
**URL:** https://arxiv.org/abs/2607.12450

---

## Headline

RINO (RGB In, RGB Out) unifies all visual perception tasks under a single RGB-to-RGB image editing framework, enabling one backbone to handle depth estimation, segmentation, surface normals, and keypoint prediction without task-specific fine-tuning — by analogy with how text provides a universal interface for language models.

---

## Key Findings

- All visual modalities (depth maps, segmentation masks, surface normals, keypoint heatmaps) can be losslessly rendered as RGB images and decoded back with standard image operations.
- Converting every vision task to an RGB-to-RGB problem allows a single image editing backbone to perform dense perception with no task-specific architecture or fine-tuning.
- RINO achieves competitive zero-shot performance on segmentation, monocular depth estimation, surface normal prediction, and 3D keypoint localization using a single set of weights.
- The formulation is analogous to how language models operate: text is the universal token type for NLP; RGB is the proposed universal pixel type for vision.
- Parameter sharing across modalities provides implicit regularization and enables transfer: gains on one task improve related tasks when they share similar RGB encodings.

---

## Methodology

RINO establishes a one-to-one correspondence between task outputs and RGB images:

1. **Depth maps:** Normalized to [0, 255] and encoded as single-channel grayscale stored in the red channel; green and blue carry uncertainty.
2. **Segmentation masks:** Each semantic class receives a fixed RGB color assignment; instance masks use hue-shifted variants.
3. **Surface normals:** The XYZ components of the unit normal vector are linearly mapped to RGB triplets.
4. **Keypoint heatmaps:** Gaussian confidence heatmaps for each joint are encoded as separate color channels.

A text instruction specifies the desired output type (e.g., "Convert to depth map" or "Generate segmentation mask"). The backbone — trained on natural image editing — processes the conditioning image and instruction without modification, producing an RGB output that is decoded into the task representation.

---

## Technical Details

- **Backbone:** Generic RGB image editing model trained on diverse image transformation tasks (style transfer, colorization, inpainting, etc.); no architecture changes.
- **Task encoding:** Task type is specified via natural language instruction; no task-specific heads, adapters, or tokens.
- **RGB-to-task decoding:** Linear or affine inverse maps convert output RGB values to task-specific ranges (e.g., depth in meters, normal unit vectors, class logits).
- **Zero-shot evaluation:** RINO is evaluated directly from the pretrained editing backbone without any fine-tuning on perception benchmarks.
- **Failure modes:** Color ambiguities in segmentation (two instances with visually similar RGB assignments) and quantization artifacts in high-dynamic-range depth maps are noted as open challenges.

---

## Experimental Findings

- Monocular depth estimation (NYU-Depth-v2): RINO achieves zero-shot performance within 4% AbsRel of DepthAnything v2, which is fine-tuned on the benchmark.
- Semantic segmentation (ADE20K): RINO achieves 38.7 mIoU zero-shot, compared to 47.2 for a fine-tuned SegFormer-B5.
- Surface normal estimation (ScanNet): Mean angular error of 11.3° zero-shot, compared to 9.1° for a task-specific model.
- 3D keypoint localization (H3WB): Within 12% PCK of a task-specific model at zero-shot, with no body-specific fine-tuning.

The zero-shot gap narrows significantly when RGB encoding quality is high and the backbone has seen similar transformations during pretraining.

---

## Ablations and Interpretation

- Replacing RGB encoding with task-specific tokenizations (e.g., direct depth scalars) and re-training from scratch outperforms RINO on individual benchmarks but requires separate models per task.
- Text instruction quality matters: vague instructions ("make it look different") degrade performance, while specific instructions ("generate a monocular depth map with near objects in red") achieve the best results.
- The parameter sharing benefit: training RINO jointly on three tasks (depth + normals + segmentation) outperforms separate single-task models by 1–2% on each task, suggesting mutual regularization.

---

## Reference

Timing Yang et al. **Let RGB Be the Language of Vision.** arXiv:2607.12450, July 2026. https://arxiv.org/abs/2607.12450
