# CPDA: Class-Conditional Path Distribution Alignment for Unsupervised Time-Series Domain Adaptation

**arXiv:** 2608.09193
**Authors:** Felix Ott, Christopher Mutschler
**Venue:** arXiv preprint
**Date:** August 10, 2026
**URL:** https://arxiv.org/abs/2608.09193

---

## Summary

Unsupervised time-series domain adaptation (UTS-DA) must bridge distribution shifts induced by different users, sensors, or temporal dynamics without target labels. Prior methods align only global feature marginals, discarding class structure during transfer. CPDA is a non-adversarial, discrepancy-based framework that aligns source and target class-conditional latent path distributions via a composite signature-spectral kernel capturing temporal path structure, frequency information, and semantic features. Source labels and target soft pseudo-labels enable class-preserving alignment. CPDA achieves state-of-the-art performance on 13 UTS-DA benchmarks against 30 competing methods.

## Problem

Time-series domain adaptation arises whenever a classifier trained in one setting (e.g., one user, one sensor model, one recording environment) must be deployed in another without access to target labels. Examples include:
- Activity recognition across different users' movement patterns
- Fault detection across different machine configurations
- Health monitoring across different wearable devices

The challenge is that time-series data has structure that static feature alignment methods miss:
- **Temporal dynamics:** the sequential ordering of features carries information
- **Class-conditional structure:** different classes have different distributional signatures; global alignment can mix class boundaries
- **Frequency-domain content:** periodic signals encode important discriminative information in their spectral components

## Key Idea: Composite Signature-Spectral Kernel

CPDA defines a novel kernel over latent time-series paths that simultaneously measures:
1. **Pooled semantic features:** overall distributional similarity (classic MMD component)
2. **Temporal path structure:** via path signatures — ordered polynomial statistics of the time-series trajectory
3. **Frequency-domain information:** spectral content captured via Fourier-based kernel components
4. **Low-rank path-signature dynamics:** compressed, dominant modes of temporal variation

This composite kernel provides a more complete characterization of the time-series distribution than any single component alone.

## Class-Preserving Alignment

Rather than aligning global marginals, CPDA conditions alignment on class:
- **Source:** class identity is known from labels
- **Target:** soft pseudo-labels are estimated iteratively
- Alignment is performed within each class, preserving class boundaries across the domain shift

This class-conditional approach prevents the "negative transfer" phenomenon where global alignment mixes class clusters from the source with different clusters in the target.

## Experimental Results

- **13 UTS-DA benchmarks:** including activity recognition, fault detection, and physiological monitoring datasets
- **30 baselines:** spanning discrepancy-based, adversarial, and pseudo-labeling methods
- **3 backbone architectures:** CNN, ResNet18, TCN
- **Result:** CPDA achieves consistent state-of-the-art across this diverse evaluation suite

## Why It Matters

Time-series domain adaptation is a critical practical problem in IoT, healthcare, and manufacturing, where labeling data in each new deployment context is prohibitively expensive. CPDA's signature-spectral kernel is the first approach to jointly capture temporal path structure and frequency content in a single principled discrepancy measure. The non-adversarial training avoids the instability and mode collapse issues of GAN-based alignment, making CPDA easier to train and more reproducible.

## Key Contributions

1. Composite signature-spectral kernel combining path signatures, spectral features, semantic pooling, and low-rank dynamics
2. Class-conditional alignment using source labels and target soft pseudo-labels
3. Non-adversarial discrepancy-based framework for UTS-DA
4. State-of-the-art on 13 benchmarks against 30 competitors with 3 backbones
