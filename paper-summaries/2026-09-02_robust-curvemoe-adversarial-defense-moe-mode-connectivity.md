# Robust CurveMoE: Multi-Norm Adversarial Defense for Mixture-of-Experts Models via Mode Connectivity

**arXiv:** 2608.26043  
**Authors:** Xu Zhang, Ren Wang  
**Submitted:** August 2026  
**Area:** Adversarial Robustness, Mixture-of-Experts, Mode Connectivity, ML Security

---

## Summary

Mixture-of-Experts (MoE) models achieve strong accuracy at reduced per-token compute cost, but their robustness to adversarial perturbations under multiple threat norms simultaneously has been poorly studied. This paper proposes Robust CurveMoE, which leverages mode connectivity—the existence of low-loss paths through weight space connecting independently trained solutions—to combine per-norm robust MoE models into a single model with multi-norm adversarial robustness. The approach avoids the accuracy-robustness trade-off that single-model adversarial training incurs and extends the mode connectivity paradigm from dense networks to MoE architectures.

## Problem

Standard adversarial training (AT) optimizes a model against perturbations within a single $\ell_p$ threat model (typically $\ell_\infty$). A model trained under $\ell_\infty$ threats may be vulnerable to $\ell_2$ or $\ell_1$ perturbations, and vice versa. Multi-norm robustness—simultaneous robustness against $\ell_\infty$, $\ell_2$, and $\ell_1$ attacks—requires either:
1. **Union training:** Training against the union of all three threat models simultaneously, which is computationally expensive and typically degrades accuracy relative to single-norm AT.
2. **Ensemble or averaging:** Combining predictions from multiple single-norm robust models at inference, which multiplies computational cost.

For MoE models specifically, an additional challenge is that the router (gating network) introduces discrete expert selection, making the loss landscape non-smooth and complicating the application of mode connectivity, which was developed for dense networks.

## Method

**Mode Connectivity Review:** Mode connectivity (Garipov et al., 2018) shows that pairs of independently trained models $\theta_A$ and $\theta_B$ can be connected by a smooth low-loss path $\phi: [0,1] \to \Theta$ parameterized as a Bézier curve, with the path found by minimizing $\mathbb{E}_{t \sim \text{Uniform}[0,1]}[\mathcal{L}(\phi(t))]$.

**CurveMoE Algorithm:**
1. **Per-norm robust MoE training:** Train three MoE endpoint models $\theta_{\ell_\infty}$, $\theta_{\ell_2}$, $\theta_{\ell_1}$, each using adversarial training under the respective threat model.
2. **Pairwise path finding:** For each pair $(\theta_A, \theta_B)$, find a Bézier curve $\phi_{AB}$ connecting them through weight space with low clean and adversarial loss.
3. **Path sampling:** At inference (or for final model selection), sample a model $\theta^* = \phi_{AB}(t^*)$ at the $t^*$ that achieves the best trade-off across all three threat models on a held-out validation set.

The key challenge is handling MoE routing: the router's argmax expert selection creates a discontinuous computational graph. The paper resolves this by applying Gumbel-softmax relaxation during path optimization, making the path optimization differentiable, and reverting to hard routing during evaluation.

## Technical Formulation

The multi-norm robust path optimization objective is:

$$\min_{\phi} \mathbb{E}_{t \sim U[0,1]} \left[ \mathcal{L}_\text{clean}(\phi(t)) + \sum_{p \in \{\infty, 2, 1\}} \alpha_p \mathcal{L}_p^\text{adv}(\phi(t)) \right]$$

where $\mathcal{L}_p^\text{adv}(\theta) = \max_{\|\delta\|_p \leq \epsilon_p} \mathcal{L}(\theta; x + \delta, y)$ is the adversarial loss under norm $p$.

The Bézier curve $\phi: [0,1] \to \Theta$ of degree 2 is parameterized by the midpoint $\theta_c$:

$$\phi(t) = (1-t)^2 \theta_A + 2t(1-t) \theta_c + t^2 \theta_B$$

with $\theta_c$ optimized by gradient descent on the path objective.

For a 3-endpoint path connecting all three robust models, the paper uses a triangular path:

$$\phi(t_1, t_2) = t_1 \theta_{\ell_\infty} + t_2 \theta_{\ell_2} + (1-t_1-t_2) \theta_{\ell_1}, \quad t_1 + t_2 \leq 1$$

## Results

- **Multi-norm robustness:** CurveMoE achieves robust accuracy of [86.2%, 71.4%, 68.3%] under [$\ell_\infty$, $\ell_2$, $\ell_1$] attacks respectively (ResNet MoE, CIFAR-100), vs. single-norm AT baselines of [89.1%, 61.2%, 55.7%] under best-matching norm only.
- **Clean accuracy:** CurveMoE clean accuracy drops 1.2% relative to the clean accuracy of the endpoint models, much less than union adversarial training (which drops 4.8%).
- **MoE vs. dense:** The path connecting MoE endpoints remains consistently low-loss across the Gumbel-softmax relaxation, validating that mode connectivity transfers to MoE architectures.
- **Ablation on path degree:** Linear interpolation (degree 1) fails—the interpolated path passes through a high-loss region. Degree-2 Bézier curves are sufficient to find a low-loss connecting path.

## Implications

CurveMoE shows that mode connectivity is a practical tool for building multi-norm robust models without per-inference ensemble overhead. The extension to MoE architectures expands the applicability of mode connectivity from dense networks to the sparse, routing-dependent models increasingly deployed in production. The approach applies to any MoE architecture with differentiable or relaxed routing.

## Reference

Xu Zhang, Ren Wang. **Robust CurveMoE: Multi-Norm Adversarial Defense for Mixture-of-Experts Models via Mode Connectivity.** arXiv:2608.26043, August 2026.  
https://arxiv.org/abs/2608.26043
