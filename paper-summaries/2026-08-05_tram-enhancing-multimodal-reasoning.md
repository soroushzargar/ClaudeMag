# TRAM: Enhancing Multimodal Reasoning with Trajectory-Derived Auxiliary Memory

**arXiv:** 2608.01922  
**Authors:** Kang Liu, Zijing Wang, Yongkang Liu, Mengjie Zhao, Xiaocui Yang, Shi Feng, Yifei Zhang, Daling Wang  
**Date:** August 3, 2026  
**URL:** https://arxiv.org/abs/2608.01922

---

## Summary

Multimodal Large Reasoning Models (MLRMs) combine vision encoders with large language models to perform multi-step visual reasoning. While these models excel on tasks requiring visual understanding and inference, they degrade as reasoning chains lengthen: intermediate conclusions established early in the trajectory lose influence by the time later steps need them. TRAM is a training-free method that augments standard MLRM decoding with an auxiliary memory pathway derived from the reasoning trajectory itself, restoring the model's ability to integrate reasoning-derived information across extended chains.

## The Problem: Long-Chain Reasoning Degrades

Standard chain-of-thought reasoning asks a model to produce a sequence of intermediate reasoning steps before reaching a final answer. In multimodal settings, the reasoning chain transforms visual observations into task-specific relations, constraints, and intermediate conclusions. The issue is that these derived facts are represented only implicitly in the LLM's context window. As the chain grows, earlier context is progressively diluted by later tokens, and the model's attention may fail to retrieve key intermediate conclusions when needed.

Attribution analysis from the paper shows that correctness in MLRMs is not strongly predicted by visual token attribution alone—it is more closely tied to whether trajectories successfully retain and integrate reasoning-derived information at each step. This motivates a focused memory intervention rather than retraining the model or expanding the context window.

## TRAM's Method: Trajectory-Derived Auxiliary Memory

TRAM maintains a compact external memory constructed from the reasoning trajectory so far. At each decoding step, the system:

1. Extracts key reasoning-derived propositions from the trajectory up to the current point
2. Stores them in a structured auxiliary memory indexed by type (visual relation, numeric fact, logical constraint, etc.)
3. Injects the relevant memory entries as additional context for the current decoding step

Because the method operates on top of any standard MLRM without modifying its weights or architecture, it is training-free—it can be applied to deployed models as a plug-in inference-time wrapper.

## Why Training-Free Matters

Retraining large multimodal models is expensive and risks catastrophic forgetting of other capabilities. Fine-tuning on curated long-chain examples is data-hungry and may not generalize. TRAM avoids both problems by augmenting the decoding process externally, making it directly deployable on top of any existing MLRM, including proprietary models accessed via API.

## Key Contributions

1. Attribution analysis showing that long-chain MLRM failures trace to trajectory information loss, not visual grounding failures
2. TRAM: training-free trajectory memory augmentation applicable to any MLRM
3. Improved accuracy on long-chain multimodal reasoning benchmarks without fine-tuning
4. Modular architecture: can be layered onto deployed models as an inference wrapper
