# Reversal Q-Learning

### Flow-Based Offline Reinforcement Learning Without Backpropagation Through Time

**Authors:** Aditya Oberai, Seohong Park, Sergey Levine (UC Berkeley)  
**Venue:** arXiv preprint  
**arXiv:** [2606.17551](https://arxiv.org/abs/2606.17551)  
**Submitted:** June 2026

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

Offline reinforcement learning (RL) trains agents from fixed datasets of past experience without environment interaction. The challenge is learning a high-quality policy purely from logged data — a setting where generalization and value estimation are both difficult. Expressive policy classes like diffusion models (and their continuous analog, flow matching) have recently shown promise because they can capture multimodal action distributions, which is critical in tasks where multiple behaviors in the dataset are equally valid.

However, training such flow-based policies with RL is non-trivial. Standard approaches either (a) distill a separately trained value function into a diffusion policy via guided generation, or (b) fine-tune the flow directly using policy gradient through backpropagation through time (BPTT). Both have serious drawbacks: guidance-based methods do not fully exploit the value function, while BPTT is computationally expensive, numerically unstable, and prone to exploding gradients over many denoising steps.

This paper introduces **Reversal Q-Learning (RQL)**, a new off-policy RL algorithm that trains a flow policy directly from offline data. RQL avoids BPTT entirely, makes full use of a learned Q-function at every denoising step, and outperforms prior methods on standard offline RL benchmarks.

The key insight is that flow refinement steps can be treated as individual actions in an "expanded" MDP, transforming the continuous-time denoising process into a standard discrete RL problem. Off-policy training is then enabled by "reversing" flows to generate synthetic on-policy-like trajectories from the offline data.

---

## Layer 2 — Under the Hood

**Intuition behind the problem and method.**

A flow matching model generates an action $a$ from noise $z \sim \mathcal{N}(0,I)$ by integrating a learned vector field $v_\theta(x_t, t)$ from $t=0$ (noise) to $t=1$ (action). Each integration step maps $x_t \to x_{t+\Delta t}$. The final output is the denoised action $a = x_1$.

**The expanded MDP.** RQL reframes each denoising step as an action: the "state" at step $t$ is the current noisy latent $x_t$, and the "action" is the step $\Delta x_t$ taken. The reward is zero except at the final step $T$, where the agent receives $r = Q^*(s, x_1)$ — the Q-value of the original state $s$ and the generated action. This reduces flow policy training to standard Q-learning over a deterministic MDP.

**The reversal trick.** The problem is that offline data contains $(s, a)$ pairs, not full trajectories through the expanded MDP. RQL solves this by "reversing" the flow: given an observed action $a^*$, we run the ODE backwards from $t=1$ to $t=0$ to reconstruct the latent trajectory $(x_0, x_1, \ldots, x_T = a^*)$. This gives a synthetic on-policy trajectory for each offline transition, enabling standard off-policy Q-learning.

**Bias-variance reduction.** The curse of horizon in off-policy RL (exponentially large variance with trajectory length) is mitigated by a reward-weighted return estimator that uses the learned Q-function at intermediate steps rather than bootstrapping only at the end.

The resulting algorithm does not require differentiating through the denoising chain (no BPTT), yet the full flow policy benefits from the Q-function throughout the generation process.

---

## Layer 3 — The Math

**Formal foundation.**

Let $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}$ be an offline dataset. A flow matching policy defines a continuous trajectory $x : [0,1] \to \mathbb{R}^d$ via ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t; s), \quad x_0 \sim \mathcal{N}(0, I), \quad a = x_1$$

**Expanded MDP.** Discretize $[0,1]$ into $T$ steps with $\Delta t = 1/T$. Define:
- State at step $k$: $\tilde{s}_k = (s, x_{k\Delta t})$ where $s$ is the original environment state
- Action at step $k$: $\delta_k = v_\theta(\tilde{s}_k) \cdot \Delta t$ (the flow increment)
- Reward: $R_T = Q^*(s, x_1)$, $R_k = 0$ for $k < T$

The optimal Q-function for the expanded MDP satisfies:
$$Q^*(\tilde{s}_k, \delta_k) = \begin{cases} Q^*(s, x_{k+1}) & k < T-1 \\ r + \gamma \mathbb{E}_{s' \sim P}[V^*(s')] & k = T-1 \end{cases}$$

where $x_{k+1} = x_k + \delta_k$.

**Reversal.** Given $(s, a^*) \in \mathcal{D}$, reconstruct $\{x_k^*\}_{k=0}^T$ by running the reverse ODE:
$$x_{k-1}^* = x_k^* - v_\theta(x_k^*, k\Delta t; s) \cdot \Delta t, \quad x_T^* = a^*$$

This yields synthetic expanded-MDP transitions $({\tilde s}_k, \delta_k^*, {\tilde s}_{k+1})$ for training.

**Q-learning objective:**
$$\mathcal{L}_Q(\phi) = \mathbb{E}_{k, (s,a^*) \in \mathcal{D}} \left[ \left(Q_\phi(\tilde{s}_k, \delta_k^*) - y_k \right)^2 \right]$$

where $y_k$ is the Bellman target. The flow policy is updated by:
$$\mathcal{L}_\pi(\theta) = -\mathbb{E}_{k, s \in \mathcal{D}, x_0 \sim \mathcal{N}} \left[ Q_\phi(\tilde{s}_k, v_\theta(\tilde{s}_k)) \right]$$

No gradients flow through the ODE solver; only the single-step output $v_\theta$ is differentiated.

---

## Experiments & Datasets

**Datasets:** D4RL locomotion (HalfCheetah, Hopper, Walker2d) at medium, medium-replay, and medium-expert quality levels. Standard continuous control benchmarks with 1M offline transitions each.

**Baselines:** IQL (Implicit Q-Learning), TD3+BC, DQL (Diffusion Q-Learning), IDQL, SfBC.

**Results:** RQL achieves state-of-the-art or competitive performance across all 12 D4RL tasks, particularly excelling on medium-expert splits where multimodal data benefits most from an expressive flow policy. It avoids the training instability reported with BPTT-based methods like DQL while surpassing guidance-only methods.
