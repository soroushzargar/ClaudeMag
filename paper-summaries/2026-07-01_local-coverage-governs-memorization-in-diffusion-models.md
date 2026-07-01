# Local Coverage Governs Memorization in Diffusion Models

### A Sharp Theoretical Criterion Predicts When a Diffusion Model Will Memorize vs. Generalize

**Authors:** Claudia Merger, Sebastian Goldt (SISSA / ICTP, Trieste)  
**Venue:** arXiv preprint  
**arXiv:** [2606.14390](https://arxiv.org/abs/2606.14390)  
**Submitted:** June 12, 2026

---

## Layer 1 — The Big Picture

Diffusion models can memorize training data: given the right prompt or initialization, a trained model reproduces exact copies of training images. This is problematic for privacy (training data can be extracted), copyright (models may reproduce copyrighted content), and scientific understanding (we want to know what generalization diffusion models actually perform).

The field has treated memorization largely empirically — documenting when it occurs, proposing heuristics to detect it, or building datasets to probe it. A principled theoretical account of *why* and *when* memorization happens has been lacking.

This paper provides exactly that: a rigorous theoretical criterion, derived from the connection between diffusion models and kernel density estimation (KDE), that predicts whether a given training point will be memorized based solely on the local density of data in its neighborhood and the dataset size $n$.

The criterion reveals a **sharp phase transition**: in the high-dimensional limit, the set of memorized points is precisely those that lie in regions where the local data coverage is too sparse for the model to interpolate. Dense regions generalize; sparse regions memorize. The transition is local — a single model can simultaneously memorize some training points and generalize others.

---

## Layer 2 — Under the Hood

**Why KDE?** A well-known connection in the score-matching literature shows that in the limit of infinite network capacity and training steps, a diffusion model trained on $n$ samples from $p_\text{data}$ learns a score function equivalent to kernel density estimation with a Gaussian kernel at each noise level $\sigma$:

$$\hat{s}(x, \sigma) = \nabla_x \log \hat{p}_\sigma(x), \quad \hat{p}_\sigma(x) = \frac{1}{n} \sum_{i=1}^n \mathcal{N}(x; x_i, \sigma^2 I)$$

This connection is exact for linear models and a provably good approximation for overparameterized networks. The paper exploits this to analyze memorization analytically.

**The local coverage criterion.** Consider a test point $x_\text{test}$ near training point $x_j$. The model memorizes $x_j$ if the reverse diffusion process starting near the data manifold converges to $x_j$ rather than a novel interpolation. This happens when the Gaussian kernels centered at other training points $x_i \neq x_j$ do not "overlap" enough with $x_j$ — i.e., when $x_j$ is isolated in feature space.

Formally, define the local coverage of $x_j$ at scale $\sigma$ as:
$$C_\sigma(x_j) = \frac{1}{n} \sum_{i \neq j} \mathcal{N}(x_j; x_i, \sigma^2 I)$$

The criterion is: $x_j$ is memorized if $C_\sigma(x_j) \ll 1/n$ for all relevant noise scales $\sigma$, i.e., when $x_j$ has no sufficiently close neighbors.

**The phase transition.** In the high-dimensional limit ($d \to \infty$ with $n, d$ growing at constant ratio), the authors show this criterion becomes *sharp*: there is a critical coverage threshold $C^*$ such that below it, memorization occurs with probability 1, and above it, it does not. The threshold $C^*$ depends on $n$ and $d$ through a simple formula involving the intrinsic dimensionality of the data.

---

## Layer 3 — The Math

**KDE score function.** At noise level $\sigma$, the estimated score is:
$$\hat{s}(x, \sigma) = \frac{1}{\hat{p}_\sigma(x)} \cdot \frac{1}{n} \sum_{i=1}^n \frac{x_i - x}{\sigma^2} \mathcal{N}(x; x_i, \sigma^2 I)$$

**Memorization criterion.** Point $x_j$ is memorized if the ODE trajectory $\{x_t\}_{t: T \to 0}$ starting near $x_j$ converges to $x_j$ itself. This occurs iff:
$$\sum_{i \neq j} \frac{\mathcal{N}(x_j; x_i, \sigma_t^2 I)}{\mathcal{N}(x_j; x_j, \sigma_t^2 I)} \ll 1 \quad \forall t \in [0, T]$$

Since $\mathcal{N}(x_j; x_i, \sigma^2 I) / \mathcal{N}(x_j; x_j, \sigma^2 I) = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$, this simplifies to:
$$\sum_{i \neq j} e^{-\|x_i - x_j\|^2 / 2\sigma^2} \ll 1$$

In $d$ dimensions with typical inter-point distance $\|x_i - x_j\| \approx \sqrt{2d}$ (for standard Gaussian data), the exponent is $-d/\sigma^2$. The critical noise scale where the transition occurs is $\sigma^* \sim \sqrt{d / \log n}$.

**Sharp transition in high dimensions.** For data distributed on a $k$-dimensional manifold embedded in $\mathbb{R}^d$, the local ball of radius $\sigma$ around $x_j$ contains on average $n \cdot (\sigma / R)^k$ training points (where $R$ is the manifold scale). Memorization occurs when this count is $O(1)$, i.e., when:
$$\sigma < \sigma^* \sim R \cdot n^{-1/k}$$

This is a sharp function of $n$: as $n$ increases, the memorization region shrinks with rate $n^{-1/k}$. Points on the manifold that are locally well-covered (high intrinsic density) generalize; outliers and boundary points memorize.

---

## Experiments & Datasets

The theoretical predictions are validated on:
1. **Synthetic Gaussian mixture models** in $d \in \{10, 50, 200\}$ dimensions with controlled inter-point distances.
2. **CIFAR-10** — empirically checking which training images are reproduced by a trained DDPM; the theory correctly identifies memorized samples as those with few near neighbors in feature space.
3. **CelebA (faces)** — rare face configurations are memorized; common attributes generalize.

The sharp transition is clearly visible in all experiments, with the empirical threshold matching the theoretical prediction to within a multiplicative constant.
