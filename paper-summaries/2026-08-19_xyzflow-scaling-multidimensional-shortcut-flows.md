# XYZFlow: Scaling Multidimensional Shortcut Flows for Efficient Generative Modeling

**arXiv:** 2608.12276  
**Date:** August 12, 2026  
**Authors:** Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Weiyang Liu  
**Affiliations:** CUHK; Westlake University; Johns Hopkins University  
**URL:** https://arxiv.org/abs/2608.12276

---

## Summary

Diffusion models produce high-quality images but require many iterative sampling steps, making inference expensive. The dominant approach to accelerating them — model distillation into few-step samplers — is a challenging process that inherits the quality ceiling of the teacher model and requires careful training. XYZFlow offers a different path: instead of distilling an existing slow model, it redesigns the generative framework itself so that high-quality generation is achievable in fewer function evaluations from the start.

## Core Idea: Multidimensional Flow Conditioning

Standard flow matching learns a mapping from noise to data along a single temporal dimension (the denoising trajectory from $t=T$ to $t=0$). This single-dimensional path has limited expressivity: the model must fit a diverse data distribution with trajectories that cannot condition on the structure of the trajectory being traversed.

XYZFlow extends flow matching to two additional dimensions:

- **Temporal (X):** Standard denoising time $t$
- **Historical (Y):** Conditioning on the denoising history — past states along the trajectory — enabling non-Markovian transitions that exploit accumulated context
- **Spatial (Z):** Sequential patch generation, where each spatial patch's flow path is conditioned on already-generated patches

The key insight is that richer conditioning reduces trajectory ambiguity. When the model knows what happened at earlier steps and in neighboring spatial regions, the uncertainty about the next step decreases and the trajectory can be made straighter, enabling larger step sizes without quality loss.

## Connection to Autoregressive Models

The paper establishes a theoretical connection between autoregressive image models and implicit flow straightening: autoregressive models can be interpreted as learning the maximally straight flow given maximal historical context (all previous tokens). XYZFlow generalizes this insight to continuous flows, interpolating between the zero-context (standard diffusion) and full-context (autoregressive) extremes.

## Key Results

- Higher quality per function evaluation compared to distilled one/few-step diffusion samplers
- Monotonic quality improvement as conditioning dimensions are added (X < XY < XYZ)
- Unlike distillation, XYZFlow models can be trained from scratch without a teacher — the multidimensional conditioning provides the expressivity needed for efficient generation

## Significance

XYZFlow shows that the expressivity bottleneck of flow-based generative models can be addressed through conditioning structure rather than increasing the number of steps or distilling from larger models. This opens the door to training inherently efficient generative models without depending on the quality of a pretrained teacher.
