# ENGINE: Endogenous Variational MultiScale Optimization for Zeroth-Order LLM Fine-Tuning

**Authors:** Zhuoli Ouyang, Bowen Li, Chen Hao, Yejin Choi  
**arXiv:** 2611.03293  
**Categories:** cs.LG  
**Submitted:** November 2, 2026  
**URL:** https://arxiv.org/abs/2611.03293

---

## Headline

ENGINE introduces Variational MultiScale decomposition from computational fluid mechanics into zeroth-order LLM fine-tuning, using a low-rank coarse gradient augmented with a closed-form subgrid correction to achieve a 2.61x speedup over prior ZO methods at zero additional memory overhead.

## Key Findings

- ENGINE achieves up to 2.61x speedup in forward passes versus prior zeroth-order baselines (including LOREN) with no additional peak memory overhead.
- The closed-form subgrid correction reintroduces unresolved curvature from the full parameter space into the low-rank coarse update, without additional forward passes.
- ENGINE consistently outperforms state-of-the-art ZO baselines on NLP fine-tuning benchmarks including SuperGLUE, MMLU, and instruction-following evaluations.
- The VMS decomposition reduces gradient variance by decoupling large-scale and small-scale curvature contributions, explaining the faster convergence.

## Problem

Zeroth-order (ZO) optimization enables LLM fine-tuning without backpropagation, making it practical on hardware with limited GPU memory. The two-point ZO gradient estimator requires only two forward passes per update, with memory comparable to inference. However, estimating gradients in the full high-dimensional parameter space is extremely noisy, and existing methods that restrict perturbations to a low-rank subspace (e.g., MeZO, LOREN) lose curvature information from the complement of that subspace. This leads to biased gradients that slow convergence, requiring more forward passes to reach a given accuracy level.

## Background: Variational MultiScale Methods

The Variational MultiScale (VMS) framework was developed in computational fluid dynamics to handle turbulence: it decomposes a solution space into a coarse (resolved) component and a fine-scale (subgrid) residual. The coarse component is updated with the standard numerical method, while the fine-scale contribution is accounted for via a correction term derived analytically from the coarse solution. This avoids the need to directly simulate the fine-scale behavior while still incorporating its effect on the coarse dynamics.

ENGINE adapts this decomposition to the parameter space of an LLM.

## ENGINE Architecture

Let theta be the model parameters. ENGINE decomposes perturbations as:
- Coarse component: z_c in a low-rank subspace U (dimensions d x r, r << d)
- Fine-scale residual: z_f in the complement U_perp

The ZO gradient estimate is computed on z_c (using 2 forward passes). The subgrid correction delta_f is then derived from the coarse gradient via a closed-form expression that approximates the projection of the full-space Hessian-vector product onto U_perp. This correction requires no additional forward passes — it is computed from the coarse gradient and a precomputed curvature approximation (a diagonal Hessian estimate maintained as a running average).

The final update combines the coarse gradient and the subgrid correction:
$$\theta \leftarrow \theta - \eta \left( \hat{g}_c + \delta_f \right)$$

where the subgrid correction delta_f reintroduces the missing curvature from the unresolved subspace.

## Implementation

- The low-rank basis U is updated periodically (every K steps) via a lightweight SVD of recent gradient directions.
- The diagonal Hessian estimate is maintained as an exponential moving average, requiring O(d) memory — negligible compared to the model parameters.
- ENGINE is a drop-in replacement for the ZO gradient estimation step and is compatible with any ZO optimizer.

## Experimental Results

**SuperGLUE (GPT-J-6B fine-tuning):**
- ENGINE matches or exceeds full fine-tuning accuracy on 7 of 8 tasks while using zero backpropagation.
- ENGINE requires 2.61x fewer forward passes than LOREN to reach the same accuracy.

**MMLU (5-shot, Llama-3-8B):**
- ENGINE achieves 72.3% accuracy, vs. 69.1% for LOREN and 70.8% for MeZO with the same forward-pass budget.

**Instruction following (Alpaca Eval):**
- ENGINE win-rate vs. LOREN: 61.4%, confirming quality improvements in open-ended generation.

**Memory:** Peak memory overhead of ENGINE vs. MeZO: 0% — the diagonal Hessian estimate fits in the memory already used by the optimizer state.

## Significance

ENGINE shows that tools from computational physics can be meaningfully adapted to deep learning optimization. The VMS decomposition provides a principled way to account for curvature beyond the low-rank approximation without additional compute, addressing the fundamental trade-off between memory efficiency and gradient quality in ZO fine-tuning. The result is relevant for any setting where backpropagation is unavailable or prohibitively expensive, including fine-tuning on-device or through black-box API access.
