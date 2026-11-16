# When and How SignSGD Outperforms SGD: A Functional Scaling Law Perspective

**Authors:** Zilin Wang, Binghui Li, Jia-Nan Wang, Lean Wang, Jinbo Wang, Lei Wu  
**arXiv:** 2611.16560 (stat.ML, cs.LG)  
**Submitted:** November 2026  
**Venue:** NeurIPS 2026 Spotlight  

---

## Headline

A new functional scaling law for SignSGD proves that gradient noise induces an effective diagonal preconditioner, explaining—for the first time rigorously—when and why sign-based optimisers dominate SGD on hard learning tasks and large batches.

---

## Key Findings

- SignSGD's advantage over SGD stems from **noise-induced preconditioning**: gradient noise applied through the sign operation acts as an approximate $\text{diag}(H)^{-1/2}$ preconditioner in noise-dominated regimes.
- SignSGD learns the signal **faster** but forgets noise more **slowly** than SGD.
- SignSGD strictly dominates SGD on **hard tasks** (high power-law decay exponent) and at **large batch sizes**, where its superior per-step progress outweighs its slower noise forgetting.
- On easy tasks, both optimisers reach comparable data efficiency; SignSGD remains preferable at large batch sizes because it converts more data into progress per gradient evaluation.

---

## Background and Motivation

SignSGD—and its momentum variant Adam—replaces the gradient with its element-wise sign:

$$\theta_{t+1} = \theta_t - \eta_t \cdot \text{sign}(g_t)$$

Empirically, Adam dominates SGD for training large language models. The standard explanation invokes adaptive learning rates or gradient clipping, but these explanations are informal or model-specific. The theory of *why* sign-based updates outperform gradient descent has remained elusive.

The authors seek a *functional* explanation: one that describes the entire loss trajectory, not just the final loss, and that ties the comparison to task properties (difficulty) and training conditions (batch size, noise level).

---

## Theoretical Framework: Functional Scaling Laws

### Setting

The analysis is conducted in power-law kernel regression, a tractable surrogate for neural network learning. The target function $f^*$ has coefficients decaying as $k^{-\beta}$ along eigenfunctions of the kernel, where $\beta > 1$ controls task difficulty. The training data has $n$ samples with label noise variance $\sigma^2$.

### Functional Scaling Law for SGD

The authors first derive (building on prior work) the functional scaling law for SGD:

$$\mathcal{L}_{\text{SGD}}(t) = \sum_{k} \bigl[(f^*_k)^2 \cdot \mathcal{S}(k, t) + \sigma^2 \cdot \mathcal{N}(k, t)\bigr]$$

where $\mathcal{S}(k, t)$ is the signal-learning kernel and $\mathcal{N}(k, t)$ is the noise-accumulation kernel, both depending on the learning rate schedule $\{\eta_s\}_{s=1}^t$.

### Functional Scaling Law for SignSGD

The key technical contribution is deriving the analogous law for SignSGD. The sign operation introduces two effects:

1. **Effective rescaling:** The step size in direction $k$ becomes $\eta_t / \|g_t\|_1 \cdot d$ approximately, where $d$ is the parameter dimension—this acts as a normalisation.
2. **Noise-induced preconditioning:** When gradient noise is non-negligible, the sign of a noisy gradient points in a direction biased toward high-curvature components. The expected update approximates a preconditioned gradient:

$$\mathbb{E}[\text{sign}(g_t + \xi_t)] \approx \frac{2}{\sqrt{2\pi\sigma^2_k}} \cdot g_{t,k}$$

per component $k$, where $\sigma^2_k$ is the noise variance in direction $k$. When $\sigma^2_k \propto \lambda_k$ (the eigenvalue / curvature), this becomes $\frac{2}{\sqrt{2\pi}} \cdot \lambda_k^{-1/2} \cdot g_{t,k}$—precisely a $\text{diag}(H)^{-1/2}$ preconditioner.

The functional scaling law for SignSGD is:

$$\mathcal{L}_{\text{Sign}}(t) = \sum_{k} \bigl[(f^*_k)^2 \cdot \tilde{\mathcal{S}}(k, t) + \sigma^2 \cdot \tilde{\mathcal{N}}(k, t)\bigr]$$

with the signal-learning kernel $\tilde{\mathcal{S}}$ dominating $\mathcal{S}$ (faster signal learning) and the noise kernel $\tilde{\mathcal{N}}$ also dominating $\mathcal{N}$ (slower noise forgetting).

### When Does SignSGD Win?

The comparison reduces to whether the faster signal learning more than compensates for the slower noise forgetting, which depends on the signal-to-noise ratio at the task level:

$$\text{SNR}(\beta, B) = \frac{\sum_k (f^*_k)^2}{\sigma^2 / B}$$

where $B$ is batch size. SignSGD wins when SNR is **low** (hard task, small batch size relative to noise) or batch size is **large** (noise per gradient step decreases, amplifying the preconditioning benefit). The paper derives the exact phase boundary as a function of $\beta$, $B$, and $\sigma^2$.

---

## Experiments

The authors validate predictions in three settings:

1. **Synthetic kernel regression:** Exact agreement with the functional scaling law predictions; the phase transition between SGD-wins and SignSGD-wins matches theory.
2. **Neural network training:** Transformer language models trained on character-level and word-level tasks. The difficulty proxy $\beta$ (estimated from the singular value spectrum of the embedding matrix) correctly predicts which optimizer wins.
3. **Large-batch training:** At batch size 4096+, SignSGD consistently dominates SGD, as predicted, across all difficulty levels tested.

---

## Significance

This work provides the first principled theoretical account of Adam/SignSGD's superiority over SGD for large-model training. The functional scaling law framework is more informative than final-loss comparisons: it predicts how quickly each optimizer learns signal versus accumulates noise throughout training, enabling practitioners to select optimizers based on task difficulty and batch size. The noise-induced preconditioning mechanism also suggests that methods that explicitly estimate diagonal Hessian curvature (like AdaFactor) should be understood through the same lens.
