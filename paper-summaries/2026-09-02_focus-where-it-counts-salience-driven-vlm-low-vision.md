# Focus Where It Counts: A Salience-Driven Vision-Language Model for Low Vision Assistance

**arXiv:** 2608.28218  
**Authors:** Jiazhao Liang, Hao Huang, Shuaihang Yuan, et al. (11 authors total)  
**Submitted:** August 28, 2026  
**Venue:** ECCV 2026  
**Area:** Vision-Language Models, Assistive Technology, Accessibility, Computer Vision

---

## Summary

General-purpose vision-language models (VLMs) generate captions that are comprehensive but do not prioritize scene elements according to their functional importance for users with blindness or low vision. This paper proposes a salience-driven captioning framework that trains a VLM—Salience-LLaVA—to produce captions in which scene elements are mentioned in order of their relevance for human-centered assistance. The framework is supported by three newly curated salience-annotated datasets.

## Problem

Assistive technologies for persons with blindness or low vision (BLV) increasingly leverage VLMs to provide verbal scene descriptions. However, standard VLMs are optimized for general captioning metrics (CIDEr, BLEU) that reward comprehensive coverage rather than ordering information by functional priority.

For a BLV user navigating a street, what matters is: (1) immediate hazards such as obstacles and traffic, (2) wayfinding landmarks, (3) relevant people or objects for social interaction. A caption that begins with "A blue sky with cumulus clouds..." before mentioning an oncoming cyclist is actively harmful. VLMs have no mechanism for ordering information according to user-centered salience.

## Method

**Salience-aware datasets:** The paper curates three datasets with object-level salience annotations:
- **Salience COCO:** MSCOCO images annotated with human-centered salience rankings; annotators were instructed to rank objects by their importance for a BLV person's navigation, interaction, and safety.
- **Salience Flickr:** Flickr images with the same annotation protocol.
- **Salience VizWiz:** Images from the VizWiz dataset (BLV users' own photos) with task-specific salience rankings derived from crowdsourced answers.

**Salience-LLaVA:** A VLM fine-tuned on the salience-aware datasets to generate captions in which high-salience elements appear early. The architecture extends LLaVA with:
1. A salience scoring module that predicts a scalar salience score for each detected object bounding box.
2. A caption generation module that conditions on the ranked object list, ensuring the LM attends to high-salience objects when producing the first tokens.

The training objective combines standard cross-entropy on the caption sequence with a ranking loss on the salience order, penalizing captions in which low-salience objects are mentioned before high-salience ones.

## Technical Formulation

Let $O = \{o_1, \ldots, o_n\}$ be the set of detected objects in an image with salience scores $s_i \in [0, 1]$ (higher = more salient). The salience ranking defines a partial order $\pi$ over objects. The combined training loss is:

$$\mathcal{L} = \mathcal{L}_\text{CE}(y, \hat{y}) + \lambda \mathcal{L}_\text{rank}(\pi, \hat{\pi})$$

where $\mathcal{L}_\text{CE}$ is standard captioning cross-entropy, $\hat{\pi}$ is the implicit ordering derived from the first appearance of each object in the generated caption, and $\mathcal{L}_\text{rank}$ is a pairwise ranking loss (e.g., RankNet) over mismatched object pairs.

At inference, the model generates captions autoregressively, with object salience scores prepended as soft prompts that bias the attention toward high-salience regions in the early token positions.

## Key Results

- **Salience ordering:** Salience-LLaVA places the highest-salience object in the first mention position for 84.3% of test images (vs. 51.2% for LLaVA-1.6 and 58.7% for GPT-4V-captioning baseline).
- **Caption quality:** SPICE and CIDEr scores improve modestly over the base LLaVA, indicating that salience ordering does not come at the cost of overall coverage.
- **User study:** BLV users rating captions on a 5-point helpfulness scale rate Salience-LLaVA 4.2/5 vs. 3.1/5 for standard LLaVA. The most common positive feedback was that important information appeared "right away" without requiring users to listen through the full description.
- **VizWiz generalization:** Fine-tuned on Salience COCO and Salience Flickr, the model transfers to VizWiz with a 12.3 pp improvement in salience ordering F1.

## Implications

Salience-driven captioning is a simple but high-impact adaptation of VLMs for accessibility. The framework applies to any object-detection-capable VLM and requires only salience-annotated training data, not architectural changes to the LM backbone. The three datasets released with the paper provide a benchmark for evaluating the salience ordering of future VLMs.

## Reference

Jiazhao Liang, Hao Huang, Shuaihang Yuan, et al. **Focus Where It Counts: A Salience-Driven Vision-Language Model for Low Vision Assistance.** arXiv:2608.28218, ECCV 2026.  
https://arxiv.org/abs/2608.28218
