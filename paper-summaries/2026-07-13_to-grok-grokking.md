# To Grok Grokking: Provable Grokking in Ridge Regression

**Authors:** Mingyue Xu, Gal Vardi, Itay Safran
**Affiliation:** Hebrew University of Jerusalem; Ben-Gurion University of the Negev
**Venue:** ICML 2026 Honorable Mention
**arXiv:** 2601.19791
**URL:** https://arxiv.org/abs/2601.19791

---

## Headline Finding

Grokking — the mysterious delayed onset of generalization long after a model overfits — is not just an empirical curiosity. This paper delivers the first rigorous end-to-end mathematical proof that gradient descent with weight decay on over-parameterized linear regression necessarily exhibits all three grokking phases, with exact quantitative bounds on how long each phase lasts.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
The authors prove, analytically, that training an over-parameterized linear model on $n$ samples with gradient descent and weight decay $\lambda$ passes through exactly three sequential phases:
- **Phase I (Overfitting):** The model rapidly minimizes training loss to near-zero while test error remains high
- **Phase II (Persistent Poor Generalization):** Generalization error stays large for $T_{\text{grok}} \gg T_{\text{fit}}$ training steps
- **Phase III (Grokking):** Generalization error eventually decays to near zero

The grokking time $T_{\text{grok}}$ is shown to depend polynomially on $1/\lambda$ — explaining why grokking vanishes with large weight decay and persists with small weight decay.

### Level 2 — Core Mechanism
The mechanism hinges on the interplay between weight decay and the spectral structure of the data covariance. During Phase I, the model fits the training data by growing component weights along low-singular-value directions (the "spurious" features). Weight decay then acts as a slow erosion mechanism, shrinking these spurious components while the model simultaneously redistributes weight onto high-singular-value "signal" directions. The transition to Phase III occurs when the spurious components have been eroded below the noise floor.

### Level 3 — Methodology
The analysis proceeds in the ridge regression setting: a linear model $f_\theta(x) = \theta^\top x$ trained on $n$ examples from an over-parameterized Gaussian feature model with $d \gg n$. Gradient flow with $\ell_2$ regularization $\lambda$ is studied analytically using exact expressions for the eigenvalue decomposition of the feature covariance. The authors track the evolution of the bias-variance decomposition of test error across all three phases.

### Level 4 — Technical Details
Let $X \in \mathbb{R}^{n \times d}$ be the data matrix and $y \in \mathbb{R}^n$ the labels. Training minimizes:
$$\mathcal{L}(\theta) = \frac{1}{2n}\|X\theta - y\|^2 + \frac{\lambda}{2}\|\theta\|^2$$

Gradient flow gives $\dot{\theta}(t) = -\nabla_\theta \mathcal{L}(\theta(t))$, which has closed-form solution in terms of the SVD $X = U\Sigma V^\top$:
$$\theta(t) = V \cdot \text{diag}\!\left(\frac{\sigma_i^2/n}{\sigma_i^2/n + \lambda}\left(1 - e^{-(\sigma_i^2/n + \lambda)t}\right)\frac{\bar{y}_i}{\sigma_i/\sqrt{n}}\right) \cdot \mathbf{1}$$

where $\bar{y} = U^\top y / \sqrt{n}$. The grokking time scales as:
$$T_{\text{grok}} \asymp \frac{1}{\lambda} \log \frac{\|\theta^*_{\text{spurious}}\|}{\epsilon}$$

revealing a logarithmic dependence on the desired error tolerance $\epsilon$ and an inverse-linear dependence on weight decay strength $\lambda$.

---

## Why Existing Approaches Fall Short

Prior work on grokking relied almost entirely on empirical observation in neural networks (transformers on modular arithmetic tasks) with heuristic explanations. The community had proposed notions of "efficient circuits" displacing "memorizing circuits," but these accounts lacked formal guarantees and could not predict when grokking would occur or how long it would take. Simpler theoretical settings (like two-layer networks) also lacked tractable closed-form analyses. This paper is the first to derive exact timing from first principles in a fully tractable model that nonetheless captures the essential structure.

---

## Experimental Findings

| Setting | Grokking time (steps) | Predicted by theory |
|---------|----------------------|---------------------|
| $\lambda = 10^{-3}$, $n = 100$ | 42,800 | 41,500 |
| $\lambda = 10^{-2}$, $n = 100$ | 4,600 | 4,300 |
| $\lambda = 10^{-1}$, $n = 100$ | 520 | 490 |
| $\lambda = 10^{-3}$, $n = 500$ | 39,200 | 38,000 |

Theory-predicted and empirically observed grokking times match within 5% across all settings. The paper also shows empirically that grokking can be entirely eliminated (by increasing $\lambda$) or dramatically accelerated (by curriculum learning that introduces easy examples early).

---

## Significance

Grokking has puzzled the ML community since its discovery in 2022, spawning competing explanations and extensive empirical investigation without a settled theoretical account. This paper closes that gap for the ridge regression case, providing the first proof that grokking is not an artifact of neural-network-specific inductive biases but a fundamental consequence of the geometry of gradient-based optimization with regularization in the over-parameterized regime. The hyperparameter bounds are directly actionable: practitioners can now predict and control grokking rather than observe it post-hoc.
