# Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models

**arXiv:** 2609.27166  
**Authors:** Not specified in available metadata  
**Date:** September 2026

---

## One-Line Summary

Hierarchical Bayesian analysis of the DeepSeek-R1-Distill family reveals that solve probability decays exponentially with problem hardness while the characteristic scale and output length both grow as power laws with model size—providing the first joint quantitative account of capability and efficiency scaling at inference time.

---

## Problem

Large reasoning models (LRMs) achieve impressive performance by generating extended chain-of-thought traces before answering, spending more test-time compute on harder problems. Two key dimensions—capability (can the model solve the problem?) and efficiency (how many tokens does it take?)—both depend on problem difficulty and model size, but how they jointly scale has not been characterized quantitatively. Without such a characterization, it is difficult to predict the compute required to solve problems of a given difficulty, or to compare model families on a common scale.

---

## Why Existing Approaches Fall Short

- **Single-metric evaluations** (accuracy at fixed token budget) conflate capability and efficiency, making it impossible to separate the ability to solve from the cost of solving.
- **Average accuracy across difficulties** masks the exponential decay with hardness; a model that solves all easy instances and none of the hard ones looks the same as one that solves both at intermediate rates.
- **Single model-size comparisons** cannot reveal how scale determines the capability-efficiency frontier.
- **Simple regression** on token counts fails to account for the hierarchical structure of variation across models, problem classes, and difficulty levels.

---

## Core Method

**Hierarchical Bayesian Model.** The paper models three quantities jointly:

1. **Solve probability** $p(x, m)$ as a function of instance size $n$ (proxy for hardness) and model parameter count $m$.
2. **Conditional output length** $L(x, m)$ given that the problem is solved.
3. **Characteristic scale** $\lambda(m)$: the instance size at which solve probability decays to $1/e$.

Four problem classes are studied: arithmetic, digit multiplication, list sorting, and path-finding—algorithmic tasks with exact verifiers and controllable difficulty.

**DeepSeek-R1-Distill Family.** Models range from 1.5B to 70B parameters, all distilled from a large reasoning model, enabling clean parameter-count comparisons within a controlled model family.

---

## Technical Formulation

Solve probability at instance size $n$ and model size $m$:
$$p(n, m) = p_0(m) \cdot \exp\!\left(-\frac{n}{\lambda(m)}\right)$$

Characteristic scale as a power law in model parameters:
$$\lambda(m) = \alpha \cdot m^\beta$$

Conditional output length (among solved instances):
$$\mathbb{E}[L \mid \text{solved}, n, m] = \gamma(m) \cdot n^{\delta}$$

where $\alpha, \beta, \gamma, \delta$ are fit from data using hierarchical Bayesian inference (Stan/MCMC), with partial pooling across problem classes to stabilize estimates at large difficulty.

The hierarchical structure:
$$\beta_k \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2), \quad k = 1, \ldots, K \text{ problem classes}$$

---

## Learning or Inference Procedure

1. **Data collection:** Evaluate each model in the DeepSeek-R1-Distill family on $\sim$500 instances per (class, size) cell; record solve/fail and token count.
2. **Model fitting:** Fit the hierarchical Bayesian model using MCMC; obtain posterior distributions over scaling exponents $\beta, \delta$ per class.
3. **Capability prediction:** Given a model size $m$ and instance size $n$, compute $\hat{p}(n, m)$ and $\hat{L}(n, m)$.

No training is performed; this is a measurement and analysis study.

---

## What the Paper Claims

1. **Exponential capability decay:** At fixed model size, solve probability decays approximately exponentially with instance size, with characteristic scale $\lambda(m) \propto m^\beta$.
2. **Power-law capability scaling:** $\lambda(m)$ grows as a power law with model parameter count, so larger models can solve proportionally harder instances.
3. **Power-law efficiency scaling:** Among solved instances, output length grows as a power law with instance size, independent of model size (up to a multiplicative constant).

---

## Experimental Findings

- Exponential decay fits are good across all four problem classes (arithmetic, digit multiplication, list sorting, path-finding).
- Characteristic scale exponent $\beta \approx 0.3$–$0.5$ depending on problem class, with partial pooling stabilizing estimates at larger sizes.
- Output length power-law exponent $\delta \approx 1.0$–$1.5$, indicating near-linear to super-linear growth with instance size.
- DeepSeek-R1-Distill-70B achieves roughly $10\times$ larger characteristic scale than the 1.5B model on arithmetic.
- Efficiency (tokens per solved instance) degrades (increases) with difficulty faster than capability degrades, implying diminishing returns to scale for the hardest instances.

---

## Ablations and Interpretation

- **Problem class differences:** Path-finding shows the steepest capability decay ($\beta$ closest to 1), while arithmetic shows the shallowest, suggesting task-specific bottlenecks.
- **Model family robustness:** The exponential-then-power-law pattern is consistent across the full model family, suggesting it is a structural property of this class of reasoning models rather than an artifact.
- **Implications for compute scaling:** Predicting solve probability for a new model size and problem difficulty requires only three parameters per class ($p_0, \alpha, \beta$), which are interpretable and fit stably.

---

## Reference

**Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models**. arXiv:2609.27166, September 2026.  
https://arxiv.org/abs/2609.27166
