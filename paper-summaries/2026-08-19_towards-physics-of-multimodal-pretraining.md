# Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes

**arXiv:** 2608.05000  
**Date:** August 5, 2026  
**Authors:** Junlin Han, Shengbang Tong, David Fan, Minghao Chen, Philip Torr, Filippos Kokkinos, Mike Lewis  
**URL:** https://arxiv.org/abs/2608.05000

---

## Summary

Multimodal pretraining — training a single model jointly on language, visual understanding, and visual generation — has achieved impressive results, but the mechanisms by which modalities interact during unified training remain poorly understood. This paper conducts a systematic empirical investigation to characterize these interactions, presenting four main findings.

## Finding 1: Knowledge Flow

The three modalities (language, visual understanding, visual generation) transfer knowledge across one another with distinct asymmetric patterns. Language knowledge flows readily into visual understanding tasks, but visual generation benefits less directly from language pretraining. Visual generation and understanding exhibit a bidirectional but asymmetric relationship: understanding benefits more from generation pretraining than the reverse. The paper disentangles these flows through controlled ablations, changing which modalities are present during pretraining and measuring cross-modal transfer.

## Finding 2: Modality Synergy

Whether modalities compete or cooperate during joint training depends on data complexity. Simple datasets lead to modality competition (one modality dominates gradient updates). Complex, diverse datasets produce synergy, where training on multiple modalities simultaneously yields better performance on each individual modality than single-modality training.

Architecturally, synergy is promoted by shared attention and normalization layers with modality-specific feed-forward networks. This design allows modalities to share high-level relational computations while maintaining modality-specific transformations.

## Finding 3: Early Unification and Vision Laziness

Unifying modalities from the earliest training stages is more effective than late alignment (training modalities separately and merging representations) or sequential training (pretraining one modality then fine-tuning on others). The paper identifies a "vision laziness" phenomenon: when visual input is introduced late in training, the model has already developed strong language priors and relies on these priors rather than learning visual representations from scratch. Early unification forces the model to develop visual representations jointly with language from the start, avoiding this collapse.

## Finding 4: Efficient Pretraining Recipes

Using the above findings, the paper derives efficient pretraining recipes that achieve strong generative and understanding performance using only 5% of the compute budget of naive joint training. These recipes are validated by training multiple 13.5B Mixture-of-Experts models on 2T tokens, demonstrating that principled design choices substantially reduce the cost of multimodal pretraining without sacrificing quality.

## Significance

This paper provides the first systematic "physics" of multimodal pretraining — a set of mechanistic principles that explain why certain design choices succeed and others fail. The practical payoff is a 20× reduction in compute cost for achieving comparable multimodal model quality.
