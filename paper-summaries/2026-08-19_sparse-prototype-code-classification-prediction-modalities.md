# Sparse Prototype Code Underlies Classification and Prediction Across Modalities

**arXiv:** 2608.15632  
**Date:** August 16, 2026  
**Authors:** Yehonatan Avidan, Daniel D. Lee, Haim Sompolinsky  
**URL:** https://arxiv.org/abs/2608.15632

---

## Summary

A central puzzle in deep learning is why representations that emerge from training on classification tasks generalize so broadly — both across tasks and across architectures. This paper identifies a universal representational geometry that appears across state-of-the-art models in vision, audio, and language processing, and characterizes it as a "Sparse Prototype Code."

## The Universal Geometry

For a classification-trained model, consider the representation space of all examples belonging to a class $c$. The paper shows that the classifier-relevant component of within-class variability is not random. Instead, it decomposes along two structured directions:

1. **The class prototype direction:** Variability component aligned with the class centroid $\mu_c$ — examples that are "more prototypical" of class $c$.
2. **Competing class directions:** Variability components aligned with the centroids $\mu_{c'}$ of competing classes — examples that are "confused" with specific other classes.

This geometry is sparse: only a small number of competing class directions have large variance components. The result is that within-class variability has a low-dimensional structure aligned with class centroids, a structure the paper calls the Sparse Prototype Code.

## Universality Evidence

The Sparse Prototype Code is found across:

- **Vision models:** ViT, ResNet, CLIP vision encoder
- **Audio models:** wav2vec, audio transformers
- **Language models:** BERT, GPT-class representations at classification layers

The paper reports consistent geometry across these diverse architectures and modalities, measured through the same spectral analysis of within-class covariance in the final representation space.

## Implications for Generalization

The Sparse Prototype Code explains several empirical phenomena in deep learning:

- **Inter-class confusion patterns:** The competing-class directions identify systematic confusions — models that confuse cats and dogs share aligned directions for these classes, even across different architectures.
- **Transferability:** Representations that score high on Sparse Prototype Code metrics transfer better to new tasks, because the within-class structure encodes semantically meaningful dimensions of variation.
- **Robustness to distribution shift:** Examples that lie along prototype directions (rather than competing-class directions) are more robust to distribution shift, since they are maximally representative of their class.

## Technical Contribution

The paper develops a spectral analysis framework for within-class covariance matrices, identifies the leading eigenvectors as prototype and competing-class directions, and shows that the eigenvalue distribution is sparse (a few large eigenvalues, many near-zero) across all tested models.

## Significance

The Sparse Prototype Code provides a unified geometric account of what good representations look like and why they arise from classification training. This is a step toward a principled theory of deep learning representations, with direct implications for designing better classification architectures and understanding model failures.
