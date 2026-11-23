# La-JEPA: Compact Robot Policies with Latent World Modeling

**Authors:** (arXiv:2611.08156 author list at paper)
**arXiv:** 2611.08156 (cs.RO, cs.LG)
**Submitted:** ~8 November 2026
**Venue:** Preprint

---

## Headline

A JEPA-based latent world model, trained from raw observations without task-specific demonstrations, produces compact robot manipulation policies that outperform a 10.4B-parameter rival on LIBERO-Plus while generalizing across a 50-task benchmark.

---

## Key Findings

- **La-JEPA** (Latent-Action JEPA) trains a latent world model using the JEPA (Joint Embedding Predictive Architecture) objective directly on raw robot observations.
- Outperforms **LeapBot-WA** (10.4B parameters) on LIBERO-Plus, a large-scale manipulation benchmark, despite using far fewer parameters.
- Achieves superior results on a 20-task subset of RoboTwin 2.0; the authors verify this advantage holds across the full 50-task benchmark.
- The world model is learned unsupervised, requiring no task-labeled demonstration data for pretraining.
- Policy conditioning on predicted latent future states improves sample efficiency during policy fine-tuning.

---

## Motivation and Problem

Robot manipulation policies trained end-to-end on demonstration data face a persistent challenge: **sample complexity**. Each new task requires substantial new data, because the policy must learn both low-level motor control and high-level task understanding from scratch.

World models offer a principled solution: learn a compact representation of environment dynamics once, then condition lightweight policies on the world model's predictions. This separates the data-intensive unsupervised representation learning phase from the data-efficient task-specific policy learning phase.

Previous JEPA-based world models for robotics (e.g., VLA-JEPA, LeWorldModel) improved on this paradigm but required either large model capacity, task-specific prompts, or separate pretrained encoders — limiting their practical compactness and generality. La-JEPA asks whether a minimal JEPA objective on raw pixels is sufficient to build a world model that generalizes across a wide manipulation task distribution.

---

## Method: La-JEPA

### JEPA Latent World Model

JEPA learns representations by predicting the latent embedding of a future state from the current state and a context signal, without decoding to pixels. For La-JEPA, the prediction target is the latent embedding of the future observation $o_{t+k}$ given the current observation $o_t$ and action sequence $a_{t:t+k}$:

$$\hat{z}_{t+k} = f_\theta(z_t, a_{t:t+k})$$

where $z_t = \phi(o_t)$ is the encoded current state and $f_\theta$ is the JEPA predictor. Training minimizes:

$$\mathcal{L}_{\text{JEPA}} = \left\|\hat{z}_{t+k} - \text{sg}(\phi(o_{t+k}))\right\|^2$$

where $\text{sg}(\cdot)$ denotes stop-gradient. This prevents representational collapse without requiring pixel reconstruction, keeping the latent space compact and semantically structured.

### Action Representation

La-JEPA conditions the predictor on a compact action embedding rather than raw joint torques. Actions are encoded by a learned action tokenizer that maps joint-space trajectories to a low-dimensional action token sequence $\tilde{a} = g(a_{t:t+k})$. This decouples the world model from specific robot kinematics, enabling transfer across embodiments.

### Policy Conditioning

The manipulation policy $\pi_\psi$ is conditioned on the world model's predicted latent future state:

$$a_t \sim \pi_\psi(a_t \mid z_t, \hat{z}_{t+H})$$

where $H$ is a prediction horizon. At inference, the world model predicts $H$ steps forward and the policy plans in latent space before executing. This look-ahead mechanism substantially improves sample efficiency during policy fine-tuning on new tasks.

---

## Technical Formulation

The combined training objective for La-JEPA is:

$$\mathcal{L}(\theta, \psi) = \mathcal{L}_{\text{JEPA}}(\theta) + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}}(\psi) + \lambda_{\text{WM}} \mathcal{L}_{\text{WM}}(\theta, \psi)$$

where $\mathcal{L}_{\text{BC}}$ is a behavioral cloning loss on task demonstrations and $\mathcal{L}_{\text{WM}}$ is an auxiliary loss that trains the policy to match predictions from the world model's forward rollouts. The world model pretraining phase minimizes $\mathcal{L}_{\text{JEPA}}$ alone; policy fine-tuning then adds the behavioral cloning terms.

The prediction error covariance across latent dimensions is:

$$\Sigma_z = \mathbb{E}\!\left[(\hat{z} - z)(\hat{z} - z)^\top\right]$$

Training monitors $\text{tr}(\Sigma_z)$ as a diagnostic: low-rank collapse (most variance in few dimensions) triggers a contrastive penalty that expands the active latent space.

---

## Experimental Findings

### LIBERO-Plus

LIBERO-Plus extends the original LIBERO benchmark to 10,030 tasks, testing generalization across diverse manipulation scenarios.

| Method | Parameters | LIBERO-Plus Success |
|---|---|---|
| LeapBot-WA | 10.4B | 81.2% |
| VLA-JEPA | 3.0B | 78.9% |
| LeWorldModel | 1.8B | 76.4% |
| La-JEPA (small) | 0.4B | 83.7% |
| La-JEPA (base) | 0.9B | 86.1% |
| La-JEPA (large) | 2.1B | 88.4% |

La-JEPA-small (0.4B) outperforms LeapBot-WA at 25× fewer parameters.

### RoboTwin 2.0

La-JEPA was evaluated on a 20-task subset chosen by a competing method and on the full 50-task benchmark. Results hold across both: La-JEPA achieves the highest average success rate on both subsets.

---

## Significance

La-JEPA demonstrates that parameter efficiency and task generalization in robot manipulation are achievable simultaneously, without large-scale pretraining on video internet data or expensive task demonstrations. The JEPA objective's pixel-free representation learning proves sufficient for building a world model that covers a 10,000-task manipulation distribution. The action tokenizer design enables embodiment transfer, suggesting La-JEPA as a foundation for multi-robot policy learning without per-robot world model retraining.
