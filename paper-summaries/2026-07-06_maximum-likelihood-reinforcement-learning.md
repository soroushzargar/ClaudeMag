# Maximum Likelihood Reinforcement Learning

### REINFORCE Secretly Optimizes the Wrong Objective — Here Is How to Fix It

**Authors:** Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, Andrea Zanette (Carnegie Mellon University)
**Venue:** ICML 2026 Oral Presentation
**arXiv:** [2602.02710](https://arxiv.org/abs/2602.02710)
**Submitted:** February 2026

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

Reinforcement learning with binary outcome feedback — where a model generates a rollout (e.g., code, a math proof, navigation steps) and receives reward 1 if it succeeds and 0 if it fails — is the dominant training paradigm for modern LLMs. The standard RL objective (REINFORCE / policy gradient) is theoretically motivated as maximizing expected reward, but the paper reveals a subtle and consequential flaw: **in sampling-based settings, REINFORCE does not maximize the true maximum likelihood over correct rollouts — it optimizes only a first-order lower-bound approximation.**

This gap is non-trivial: the true maximum likelihood objective encourages the model to concentrate probability mass on *the* correct answer, while REINFORCE's approximation allows probability to diffuse across many mediocre rollouts. The paper introduces **MaxRL (Maximum Likelihood Reinforcement Learning)**, a family of compute-indexed objectives that smoothly interpolate between REINFORCE and exact MLE as more sampling compute is allocated. MaxRL achieves consistent improvements across maze navigation, GSM8K, and large-scale Qwen3 training.

---

## Layer 2 — Under the Hood

**The REINFORCE bias.**

In supervised MLE, the gradient is $\nabla_\theta \log P(y | x)$ — directly maximizing the probability of the ground-truth label. In RL with binary reward, the environment only tells us whether a rollout $y$ is correct, not what the correct rollout is. REINFORCE estimates the policy gradient by weighting the log-probability of sampled rollouts by their rewards:

$$\nabla_\theta J_\text{REINFORCE} = \mathbb{E}_{y \sim \pi_\theta}\left[r(y) \nabla_\theta \log \pi_\theta(y|x)\right]$$

The paper shows this equals the gradient of a lower bound on the true MLE objective, not the objective itself. With only one sample per query, REINFORCE is equivalent to minimizing KL divergence in the *wrong direction*.

**MaxRL's interpolation.** With $k$ samples per query, MaxRL forms a better estimate of the MLE gradient:

$$\nabla_\theta J_\text{MaxRL}^{(k)} = \mathbb{E}_{y_{1:k} \sim \pi_\theta}\left[\frac{r(y_i)}{\sum_{j=1}^k r(y_j)} \nabla_\theta \log \pi_\theta(y_i|x)\right]$$

As $k \to \infty$, this converges to exact MLE; at $k=1$, it reduces to REINFORCE. The interpolation is monotone: more compute → better objective approximation.

---

## Layer 3 — The Math

**Formal statement.** Let $Y^+$ denote the set of correct responses for query $x$. The true maximum likelihood objective is:

$$J_\text{MLE}(\theta) = \mathbb{E}_{x}\left[\log \sum_{y \in Y^+} \pi_\theta(y|x)\right]$$

REINFORCE optimizes:

$$J_\text{REINFORCE}(\theta) = \mathbb{E}_{x, y \sim \pi_\theta}\left[r(y) \log \pi_\theta(y|x)\right]$$

The paper proves $J_\text{REINFORCE}(\theta) \leq J_\text{MLE}(\theta)$ (Jensen's inequality applied to the log-sum), with the gap controlled by the entropy of the reward-weighted distribution.

**MaxRL objective (k-sample estimator):**

$$J_\text{MaxRL}^{(k)}(\theta) = \mathbb{E}_{x, y_{1:k}}\left[\log \frac{1}{k}\sum_{i=1}^k r(y_i) \pi_\theta(y_i|x)\right]$$

This is an importance-weighted MLE estimator. The gradient is:

$$\nabla_\theta J_\text{MaxRL}^{(k)} = \mathbb{E}\left[\sum_i w_i \nabla_\theta \log \pi_\theta(y_i|x)\right], \quad w_i = \frac{r(y_i)\pi_\theta(y_i|x)}{\sum_j r(y_j)\pi_\theta(y_j|x)}$$

**Convergence:** As $k \to \infty$, $J_\text{MaxRL}^{(k)} \to J_\text{MLE}$ almost surely.

---

## Experiments & Datasets

**Tasks:**
- Toy image classification (verifying closeness to MLE)
- Maze navigation (RL benchmark)
- GSM8K (mathematical reasoning with binary correctness signal)
- Large-scale Qwen3 fine-tuning (full LLM training)

**Results:** MaxRL consistently outperforms REINFORCE and competitive baselines across all tasks. On GSM8K with Qwen3, MaxRL gains ~3–5% accuracy with the same number of gradient steps and compute. The gap scales with $k$: more sampling compute delivers better final performance.
