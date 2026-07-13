# Latent Memory Palace: Reasoning for Control as Autoregressive Variational Inference

**Authors:** Chuning Zhu, Eva Xu, Jose Barreiros, Krishnan Srinivasan, Paarth Shah, Abhishek Gupta
**Affiliation:** Paul G. Allen School of Computer Science and Engineering, University of Washington
**Venue:** arXiv preprint
**arXiv:** 2607.08724
**URL:** https://arxiv.org/abs/2607.08724

---

## Headline Finding

Language models reason flexibly by generating variable-length chains of thought before acting. Continuous control policies cannot: they must map observations directly to actions in a fixed computation graph. Latent Memory Palace closes this gap by casting control-time deliberation as autoregressive variational inference over a structured latent memory, enabling physical agents to dynamically allocate more or fewer reasoning steps depending on task complexity — without any symbolic planning module.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
Latent Memory Palace (LMP) introduces a deliberation-depth-adaptive control policy that:
- Learns to allocate variable numbers of latent "thinking steps" per decision
- Outperforms fixed-depth recurrent policies on tasks requiring multi-step reasoning in continuous domains
- Achieves 47% higher success rate on long-horizon manipulation tasks versus strongest recurrent baseline
- Scales favorably with maximum deliberation budget: performance improves monotonically as more latent steps are allowed

### Level 2 — Core Mechanism
The key idea is to model the latent state as a sequence generated autoregressively before each action is produced. At each time step $t$, the agent runs $K$ latent computation steps $z_t^{(1)}, \ldots, z_t^{(K)}$, where each $z_t^{(k)}$ is sampled from a learned prior conditioned on $z_t^{(k-1)}$ and the current observation $o_t$. The policy then reads off the final latent state $z_t^{(K)}$ to produce an action. The number of steps $K$ is itself a learned variable, controlled by a halting mechanism similar to adaptive computation in Universal Transformers.

### Level 3 — Methodology
The framework is formalized as a hierarchical variational autoencoder (HVAE) where the prior over latent sequences and the posterior given observations are both parametrized as autoregressive models:

1. **Prior:** $p_\theta(z_t^{(1:K)} \mid o_t) = \prod_{k=1}^K p_\theta(z_t^{(k)} \mid z_t^{(1:k-1)}, o_t)$
2. **Posterior:** $q_\phi(z_t^{(1:K)} \mid o_{1:T}, a_{1:T}) = \prod_{k=1}^K q_\phi(z_t^{(k)} \mid z_t^{(1:k-1)}, o_t, h_t^{\text{ret}})$
3. **Policy:** $\pi_\psi(a_t \mid z_t^{(K)})$

where $h_t^{\text{ret}}$ is a retrospective summary of future observations and rewards, used only during training to inform the posterior. The model is trained by maximizing the evidence lower bound (ELBO) on the sequence of observed trajectories.

### Level 4 — Technical Details
The ELBO objective for a trajectory $\tau = (o_1, a_1, \ldots, o_T, a_T)$:
$$\mathcal{L}(\theta, \phi, \psi) = \mathbb{E}_{q_\phi}\!\left[\sum_{t=1}^T \log \pi_\psi(a_t \mid z_t^{(K)}) - \beta \sum_{t=1}^T \text{KL}\!\left(q_\phi(z_t^{(1:K)}) \| p_\theta(z_t^{(1:K)})\right)\right]$$

The adaptive halting mechanism uses a learned binary halt variable $h_t^{(k)} \in \{0,1\}$ at each latent step, with the policy halting when $h_t^{(k)} = 1$. The halt probability is parameterized by a small MLP:
$$P(h_t^{(k)} = 1) = \sigma\!\left(f_h\!\left(z_t^{(k)}, \text{complexity}(o_t)\right)\right)$$

where $\text{complexity}(o_t)$ is an auxiliary feature measuring estimated task difficulty. At training time, the expected computation cost $\mathbb{E}[\sum_k h_t^{(k)} \cdot c]$ is added as a penalty to incentivize economy.

---

## Why Existing Approaches Fall Short

Model-based RL methods (Dreamer, TDMPC) use fixed-depth planning that requires pre-specifying a rollout horizon. They cannot adapt computation to task difficulty. Recurrent policies (LSTM, GRU, S4) maintain fixed-size hidden states updated at each time step but cannot perform multi-step deliberation before acting. Hybrid symbolic-neural planners achieve variable-depth reasoning but are brittle to distribution shift and require discrete state spaces. Diffusion-based policies (Diffuser, Decision Diffusion) perform iterative denoising in action space, but this iteration serves as trajectory optimization, not as task-level deliberation over abstract latent states. LMP is the first method to directly transfer the chain-of-thought deliberation mechanism from discrete language generation to continuous action generation in a principled variational framework.

---

## Experimental Findings

| Task | LMP | LSTM | S4 | Dreamer-V3 |
|------|-----|------|----|------------|
| Stacking (5 objects) | 71.3% | 48.5% | 52.1% | 44.2% |
| Multi-step tool use | 63.7% | 41.2% | 45.6% | 38.9% |
| Long-horizon pick-place | 78.4% | 56.8% | 61.3% | 52.7% |
| Deformable manipulation | 55.1% | 32.4% | 37.8% | 28.6% |

All results are task success rates (%) averaged over 5 random seeds and 200 evaluation episodes. LMP outperforms the next-best baseline by 16–26% across tasks, with larger gaps on tasks requiring longer multi-step reasoning.

---

## Significance

The ability to think before acting — to deliberate proportionally to task difficulty — is a defining feature of intelligent behavior that current continuous-control agents largely lack. Latent Memory Palace offers a principled route to this capability without discretizing the action space or bolting on a symbolic planner. The autoregressive variational framework is model-agnostic and can be applied to any continuous RL setting. If the adaptive deliberation approach scales to more complex embodied tasks (dexterous manipulation, whole-body control), it could substantially close the gap between the reasoning capabilities of modern LLMs and the action-execution capabilities of physical robotic systems.
