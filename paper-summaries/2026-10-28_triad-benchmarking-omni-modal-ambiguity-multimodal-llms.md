# TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models

**Authors:** Zhaolu Kang, Yidi Wang, Yile Li, Siqi Zeng, Yingjie He, Richeng Xuan, Zhichao Hu  
**Venue:** NeurIPS 2026 (Datasets and Benchmarks Track)  
**arXiv:** 2610.30841  
**Date:** October 22, 2026  
**URL:** https://arxiv.org/abs/2610.30841

---

## Summary

TRIAD is a new benchmark designed to test whether omni-modal language models genuinely integrate all three modalities—text, image, and audio—when resolving ambiguity, rather than relying on the most salient modality alone. The key insight is that existing multimodal benchmarks do not control for modality sufficiency: an image+text question can often be answered from text alone, masking the model's failure to use visual information.

TRIAD forces multi-modal integration by construction. Each item is a text–image–audio triplet paired with a question such that: (1) the complete triplet yields exactly one correct answer, and (2) removing any single modality leaves at least two consistent options. This "ambiguity-by-design" property is verified across 86 scene groups containing 327 hand-crafted queries, spanning 18 ambiguity categories.

---

## Dataset Design

- **Scale:** 327 queries in 86 scene groups, 18 ambiguity categories  
- **Modalities:** Text, image, and audio—each necessary; none sufficient alone  
- **Evaluation protocol:** Group accuracy (all queries in a scene group correct) and item accuracy  
- **Condition:** Answers evaluated with and without providing answer options

---

## Key Findings

### 1. Strong Item Accuracy ≠ Robust Scene-Level Disambiguation

Evaluated on 14 omni-modal systems (including leading proprietary and open-source models), the best model achieves high item accuracy but trails humans by over 20 points on group accuracy—the metric that measures whether a model can reliably disambiguate across an entire scene.

### 2. Audio Is Systematically Underused

When text and image are jointly present, models largely ignore audio cues. Performance drops sharply on items whose ambiguity can only be resolved via the audio modality, suggesting that current omni-modal architectures default to text–vision understanding.

### 3. Answer Options Inflate Apparent Performance

Removing multiple-choice answer options drops model group accuracy to single digits, while human group accuracy remains robust. This shows that models exploit option-level cues rather than performing genuine multimodal reasoning.

---

## Method Details

The benchmark pipeline involves:

1. **Scene construction:** Authors collect or synthesize scene groups where a shared context can be described in three modalities with overlapping but distinct information.
2. **Ambiguity verification:** Each triplet is manually verified to ensure that any two-modality combination leaves the correct answer underdetermined.
3. **Evaluation:** Models are prompted with the full triplet and scored on both item accuracy and scene-group accuracy.

---

## Significance

TRIAD exposes a fundamental gap between the claimed omni-modal capability of modern frontier models and their actual ability to fuse information across text, image, and audio when disambiguation requires all three. The benchmark provides a principled diagnostic for future model development and is released publicly with evaluation code.

---

## Citation

Zhaolu Kang, Yidi Wang, Yile Li, Siqi Zeng, Yingjie He, Richeng Xuan, Zhichao Hu.  
**TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models.**  
arXiv:2610.30841. NeurIPS 2026.  
https://arxiv.org/abs/2610.30841
