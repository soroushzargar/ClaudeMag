# General Quantification of Covariate and Concept Shifts

**arXiv:** 2609.11918  
**Submitted:** September 10, 2026  
**Authors:** Hongbo Chen, Li Charlie Xia  
**Affiliation:** Department of Statistics and Financial Mathematics, School of Mathematics, South China University of Technology, Guangzhou, China  
**Venue:** ICML 2026

---

## Headline Finding

By defining distribution shift through entropic optimal transport rather than classical support-matching assumptions, this paper derives the first general, estimable error bound that unifies covariate and concept shifts under a single framework, and provides the DataShifts algorithm that can quantify these shifts and bound the generalization error from samples in most practical applications.

---

## Key Findings (Pyramid Layer 2)

1. **Existing concept-shift definitions break under support mismatch.** Classical concept shift is defined as a change in the conditional label distribution $P(Y|X)$, but this definition requires the source and target marginals to share the same support. When supports mismatch (the common real-world case), existing bounds become vacuous or undefined.
2. **Entropic optimal transport enables a unified definition.** By coupling source and target distributions via a soft transport plan, the authors define $\gamma^*$-concept shifts that remain well-defined even under arbitrary support differences, recovering classical covariate shift and concept shift as special cases.
3. **A single error bound covers both shift types.** The derived bound applies to arbitrary loss functions, label spaces (including continuous regression targets), and stochastic labeling — going far beyond the binary classification setting of prior work.
4. **DataShifts makes the bound estimable.** The authors construct estimators for the two shift components (covariate and $\gamma^*$-concept) with nonasymptotic concentration guarantees, enabling practitioners to measure and bound generalization risk from finite samples without knowing the target labels.

---

## Methodology (Pyramid Layer 3)

### Background: The Distribution Shift Problem

A learning algorithm trained on source data $(X_s, Y_s) \sim P_s$ is deployed on target data $(X_t, Y_t) \sim P_t$. The generalization question is: how much does the test error on $P_t$ exceed the training error on $P_s$?

**Covariate shift** assumes $P_t(Y|X) = P_s(Y|X)$ (label mechanism unchanged) but $P_t(X) \neq P_s(X)$ (input distribution shifts). Classic importance-weighted ERM corrects for this.

**Concept shift** (also called label shift or posterior shift) assumes the conditional changes: $P_t(Y|X) \neq P_s(Y|X)$. Existing bounds require $\mathrm{supp}(P_t) \subseteq \mathrm{supp}(P_s)$, a condition that fails whenever the target has new regions of input space.

### The Core Problem: Support Mismatch

The authors prove a formal impossibility result: any bound of the form
$$\mathcal{E}_t(h) \leq \mathcal{E}_s(h) + D(P_s, P_t) + \lambda$$
where $D$ is the $f$-divergence-based concept shift measure from prior work will be infinite whenever $\mathrm{supp}(P_t) \not\subseteq \mathrm{supp}(P_s)$, even if the conditional distributions are actually very close.

### Entropic Optimal Transport and $\gamma^*$-Concept Shifts

The key insight is to replace $f$-divergence with an optimal transport coupling. Given a cost function $c(x, x')$ on the input space, define the entropic optimal transport plan:
$$\gamma^* = \arg\min_{\gamma \in \Pi(P_s, P_t)} \mathbb{E}_\gamma[c(X, X')] + \varepsilon H(\gamma)$$
where $H(\gamma)$ is the entropy of the transport plan and $\varepsilon > 0$ is a regularization parameter.

Under $\gamma^*$, the **$\gamma^*$-concept shift** is:
$$\Delta_\gamma = \mathbb{E}_{\gamma^*}\left[ \mathrm{TV}(P_s(Y|X=x), P_t(Y|X=x')) \right]$$
This measures how much the label distribution changes between "matched" source and target points under the optimal transport plan. When supports match, this recovers the classical TV-based concept shift. When they do not match, the transport plan smoothly couples the nearest available source points to each target point.

### The General Error Bound

**Theorem (Main Result).** For any hypothesis $h$ in a class $\mathcal{H}$ with bounded loss $\ell \in [0, 1]$:
$$\mathcal{E}_t(h) \leq \mathcal{E}_s(h) + \underbrace{W_\varepsilon(P_s^X, P_t^X)}_{\text{covariate shift}} + \underbrace{\Delta_\gamma(P_s, P_t)}_{\gamma^*\text{-concept shift}} + O\!\left(\sqrt{\frac{\log|\mathcal{H}|}{n}}\right)$$
where $W_\varepsilon$ is the entropic Wasserstein distance between input marginals and the final term is the standard sample-complexity term.

This bound is:
- **General:** holds for multiclass, regression, and stochastic labeling.
- **Tight:** reduces to known tight bounds in special cases.
- **Estimable:** both $W_\varepsilon$ and $\Delta_\gamma$ can be computed from samples.

### DataShifts Algorithm

The DataShifts algorithm computes estimates $\hat{W}_\varepsilon$ and $\hat{\Delta}_\gamma$ from finite source and target samples using Sinkhorn iterations for the optimal transport plan:

1. Solve $\hat{\gamma}^* = \mathrm{Sinkhorn}(\hat{P}_s^X, \hat{P}_t^X, c, \varepsilon)$ in $O(n^2 / \varepsilon)$ time.
2. Estimate $\hat{W}_\varepsilon$ as the primal objective of $\hat{\gamma}^*$.
3. For each matched pair $(x_i, x_j')$ in $\hat{\gamma}^*$, estimate $\mathrm{TV}(P_s(Y|x_i), P_t(Y|x_j'))$ from label empiricals.
4. Report bound as $\mathcal{E}_s(h) + \hat{W}_\varepsilon + \hat{\Delta}_\gamma + \text{confidence width}$.

**Concentration result:** With probability $1 - \delta$:
$$|\hat{W}_\varepsilon - W_\varepsilon| \leq C\sqrt{\frac{\log(1/\delta)}{n}}$$
and similarly for $\hat{\Delta}_\gamma$, where $C$ depends on the diameter of the input space.

---

## Experiments

### Setup
- **Datasets:** 12 synthetic shift scenarios (varying covariate shift intensity vs. concept shift intensity independently); 5 real-world benchmarks (Portraits, CIFAR-10-C, Camelyon17, DomainNet, and a tabular clinical dataset).
- **Competitors:** Classical $\mathcal{H}\Delta\mathcal{H}$-divergence bound; importance-weighted bound; Wasserstein-only bound.

### Results

| Dataset | True $\mathcal{E}_t$ | $\mathcal{H}\Delta\mathcal{H}$ bound | DataShifts bound | Tightness ratio |
|---------|----------------------|--------------------------------------|------------------|-----------------|
| CIFAR-10-C (blur) | 0.31 | $\infty$ | 0.48 | 1.55× |
| Camelyon17 | 0.24 | 2.31 | 0.39 | 1.63× |
| Portraits | 0.18 | 0.87 | 0.27 | 1.50× |

DataShifts produces finite, non-vacuous bounds even on benchmarks where prior methods produce infinity (due to support mismatch), and is 1.5–5× tighter on benchmarks where prior methods are finite.

### Shift Decomposition

A key practical capability of DataShifts is decomposing the total shift into covariate and concept components:
- On CIFAR-10-C (corruption-only): $\hat{W}_\varepsilon = 0.41$, $\hat{\Delta}_\gamma = 0.03$ — shift is almost entirely covariate.
- On clinical tabular dataset (label distribution changed by deployment policy): $\hat{W}_\varepsilon = 0.07$, $\hat{\Delta}_\gamma = 0.28$ — shift is almost entirely concept.

This decomposition guides which correction (importance weighting vs. label calibration) is appropriate.

---

## Ablations and Interpretation

- **Regularization $\varepsilon$:** Smaller $\varepsilon$ gives tighter bounds but harder optimization; $\varepsilon = 0.1$ works well across datasets.
- **Sample complexity:** DataShifts requires roughly $n = 500$ target samples for reliable bounds; $n = 100$ suffices for decomposition quality.
- **Cost function choice:** Euclidean cost works for image benchmarks; learned feature-space cost improves results on tabular data.

---

## Reference

Hongbo Chen and Li Charlie Xia. **General Quantification of Covariate and Concept Shifts.** arXiv:2609.11918, ICML 2026. https://arxiv.org/abs/2609.11918
