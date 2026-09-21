# AgriScope: Pixel-Grounded Multimodal Understanding for Agricultural Images

**arXiv:** 2609.20325  
**Submitted:** September 17, 2026  
**Authors:** Abderrahmene Boudiaf, Mohamad Alanssari, Irfan Hussain, Sajid Javed  
**Affiliation:** Khalifa University of Science and Technology  
**Venue:** ICML 2026 (Forty-Third International Conference on Machine Learning)

---

## Headline Finding

AgriScope is the first unified multimodal framework that jointly supports image-level, region-level, and pixel-level agricultural image understanding within a single model, combining biological-semantic encoding with dense spatial grounding and a pixel decoder; on the accompanying AgriGround benchmark (500K images, 11M samples), AgriScope raises grounded caption CIDEr from 107.9 to 118.6 and mIoU from 55.0 to 59.4 over the best prior MLLM baseline.

---

## Key Findings (Pyramid Layer 2)

1. **Agricultural image understanding is uniquely demanding.** Plant disease symptoms, pest morphologies, and crop structural features vary subtly across growth stages, lighting conditions, and camera angles. The fine-grained spatial specificity required (e.g., "early-stage powdery mildew on the third leaf from the apex") far exceeds what general-purpose MLLMs provide, which produce text descriptions without pixel-level localization.

2. **Existing MLLMs produce text-only outputs incompatible with precision agriculture.** Actionable recommendations in precision agriculture (targeted pesticide application, selective harvesting) require pixel-level region identification. Without grounded outputs, a diagnosis of "this plant has rust disease" cannot drive automated machinery or produce heatmaps for the farmer.

3. **Biological-semantic encoding captures domain-specific visual structure.** AgriScope integrates a specialized encoder pre-trained on botanical taxonomies and disease atlases that produces features aligned with biological categories (species, disease stage, morphological class). These features are richer than general vision encoders for fine-grained agricultural distinctions.

4. **AgriGround provides the first large-scale pixel-grounded agricultural instruction dataset.** At 500K images and 11M instruction-following samples, AgriGround is 30× larger than the largest prior agricultural vision dataset. Its multi-stage construction pipeline combines multimodal caption generation, phrase-level grounding, segmentation mask generation, and task-oriented instruction synthesis to produce fully annotated, task-diverse training data.

---

## Methodology (Pyramid Layer 3)

### Architecture Overview

AgriScope builds on a standard MLLM backbone (vision encoder + LLM decoder) with three additional components:

**Biological-Semantic Encoder (BSE):** A ViT-L model fine-tuned on a curated corpus of 2.3M agricultural images with hierarchical biological labels (kingdom, class, order, family, genus, species, disease type, disease stage). The BSE produces tokens $\{f_i^{\text{bio}}\}_{i=1}^{N_v}$ that encode biological category memberships alongside spatial features.

**Dense Spatial Representations (DSR):** A multi-scale feature pyramid built by combining BSE features with a frozen DINOv2 visual encoder. The pyramid provides spatially dense representations at 1/4, 1/8, and 1/16 of the input resolution, enabling both coarse and fine-grained spatial reasoning:
$$F_{\text{DSR}} = \text{FPN}\!\left(f^{\text{bio}}, f^{\text{dino}}\right) \in \mathbb{R}^{C \times H/4 \times W/4}$$

**Pixel Decoder (PD):** A mask transformer decoder that takes as input a set of query embeddings $\{q_k\}$ derived from the LLM's text outputs (phrase-level noun chunks) and predicts binary segmentation masks:
$$M_k = \text{PD}(q_k, F_{\text{DSR}}) \in \{0,1\}^{H \times W}$$

The LLM output interleaves natural language tokens with special grounding tokens `[SEG]`; when `[SEG]` appears, the last LLM hidden state at that position serves as $q_k$ for the pixel decoder.

### AgriGround Construction Pipeline

1. **Source image collection:** 500K agricultural images from public datasets (PlantVillage, iNaturalist-Agriculture, FGVC-Plant, CropDiseaseDB) plus curated web imagery.
2. **Multimodal caption generation:** GPT-4V with botanical system prompts generates grounded captions identifying diseases, pests, species, and structural features with region references.
3. **Phrase-level grounding:** Grounding-DINO and SAM2 jointly extract bounding boxes and segmentation masks for each referenced noun phrase.
4. **Task-oriented instruction synthesis:** Templates across 6 task types (disease diagnosis, pest identification, species classification, crop growth assessment, weed detection, nutritional deficiency analysis) convert grounded captions into instruction-following samples.

---

## Technical Formulation

The training objective combines a language modeling loss, a mask binary cross-entropy loss, and a mask dice loss:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$

The language modeling loss is standard next-token prediction over the instruction-following samples. The mask losses apply to predicted masks $\hat{M}_k$ versus ground-truth masks $M_k^*$:

$$\mathcal{L}_{\text{BCE}} = -\sum_{p} \left[M_k^*(p) \log \hat{M}_k(p) + (1-M_k^*(p))\log(1-\hat{M}_k(p))\right]$$

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2\sum_p M_k^*(p)\hat{M}_k(p)}{\sum_p M_k^*(p) + \sum_p \hat{M}_k(p)}$$

with $\lambda_{\text{BCE}} = 1.0$ and $\lambda_{\text{dice}} = 0.5$.

---

## Learning or Inference Procedure

1. Pre-train the Biological-Semantic Encoder on the 2.3M agricultural image corpus with hierarchical classification.
2. Train the full AgriScope model on AgriGround with frozen DINOv2 and trainable BSE, LLM adapter layers, and pixel decoder for 3 epochs.
3. Fine-tune for specific task subsets (disease detection, pest recognition) for 1 additional epoch.
4. At inference: input an image and instruction; the LLM generates text interleaved with `[SEG]` tokens; the pixel decoder produces segmentation masks for each `[SEG]` token.

---

## What the Guarantee Says

AgriScope does not provide formal guarantees. The ICML 2026 reviewers highlighted the breadth of the AgriGround evaluation: the paper reports results across all 6 task categories plus 3 cross-dataset generalization settings, providing strong empirical evidence that the improvements generalize beyond in-distribution agricultural imagery.

---

## Experimental Findings

### Benchmarks
- **AgriGround-Test:** 10K held-out samples from AgriGround across all 6 task types
- **PlantDoc (grounded):** 2,578 images across 17 crop diseases
- **iNaturalist-Agri-2K:** Fine-grained botanical recognition

### Main Results on AgriGround-Test

| Method | GCG CIDEr ↑ | GCG mIoU ↑ | Seg F1 ↑ | VQA Acc ↑ |
|--------|------------|-----------|---------|----------|
| LLaVA-1.5 | 82.3 | 43.7 | — | 71.4 |
| InternVL-2 | 97.1 | 49.8 | — | 78.9 |
| LISA | 101.4 | 52.3 | 64.1 | 74.2 |
| PixelLM | 107.9 | 55.0 | 67.8 | 76.5 |
| **AgriScope** | **118.6** | **59.4** | **73.2** | **82.7** |

GCG = Grounded Caption Generation. Seg F1 = referring expression segmentation.

---

## Ablations and Interpretation

- **Without BSE (standard ViT encoder):** CIDEr drops by 6.8 points; the standard encoder misses fine-grained disease stage distinctions.
- **Without multi-scale DSR (single-resolution features):** mIoU drops by 3.1 points; single-scale features miss small lesions and pest bodies.
- **Without AgriGround (training on GCG-only datasets):** CIDEr drops by 14.3 points; breadth of agricultural supervision is critical.
- **Pixel decoder variants:** Mask transformer decoder (+2.7 mIoU vs. FPN upsampling decoder); SAM-style decoder (+0.9 mIoU but 2× parameters).
- **AgriGround scale:** Training on 100K of 500K samples loses 4.2 CIDEr points; scaling from 250K to 500K gains 2.1 CIDEr points, suggesting continued benefit from more data.

---

## Reference

Abderrahmene Boudiaf, Mohamad Alanssari, Irfan Hussain, and Sajid Javed. **AgriScope: Pixel-Grounded Multimodal Understanding for Agricultural Images.** arXiv:2609.20325, September 2026. https://arxiv.org/abs/2609.20325. Accepted at ICML 2026.
