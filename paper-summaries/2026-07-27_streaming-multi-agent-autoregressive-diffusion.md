# Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers

**Authors:** Sicheng Mo, Yuheng Li, Ziyang Leng, Krishna Kumar Singh, Bolei Zhou  
**Venue:** arXiv:2607.21594  
**Date:** July 23, 2026  
**URL:** https://arxiv.org/abs/2607.21594

---

## Summary

Generating long-horizon, multi-perspective video simulations—as required in autonomous driving, robotics, and multi-agent game environments—demands temporal and cross-agent spatial consistency that standard video diffusion models cannot maintain. This paper introduces a streaming autoregressive diffusion framework with a **World State Register**: a shared, updatable global context mechanism that allows multiple diffusion agents to generate their respective views coherently over arbitrarily long rollouts. The approach achieves significantly improved consistency in multi-agent autonomous driving simulations compared to per-agent baselines.

## Problem

Multi-agent video generation faces three compounding challenges:

1. **Temporal drift:** Autoregressive video generation accumulates error over time—each new segment is conditioned on previously generated frames, and small inconsistencies in early frames propagate and amplify through long rollouts.

2. **Cross-agent inconsistency:** When multiple agents generate their own views independently (even of the same underlying scene), the independent diffusion processes produce mutually contradictory depictions of shared 3D world elements—objects appearing in one agent's view but missing or distorted in another's.

3. **Scalability:** Passing full multi-agent history as conditioning context to each diffusion model grows quadratically in both agents and time, making streaming generation intractable for large or long simulations.

## Method: Streaming Multi-Agent Autoregressive Diffusion

### World State Registers (WSR)
The central innovation is the **World State Register**: a compact, fixed-size set of latent vectors $\mathbf{s} \in \mathbb{R}^{M \times d}$ that encode the current global state of the simulated world, shared across all $A$ agents. The register is:
- **Aggregated** after each generation step: all agents' newly generated frames are encoded by a shared encoder and used to update the register via a cross-attention write operation.
- **Broadcast** before each generation step: the register is distributed as conditioning context to all agents' diffusion denoising processes.
- **Fixed-size:** No matter how long the rollout or how many agents are involved, the register has a constant $M$ vectors, preventing quadratic context growth.

### Autoregressive Diffusion per Agent
Each agent $a$ generates video frames autoregressively using a latent diffusion model conditioned on:
1. Its own previous $W$ generated frames (a sliding local window).
2. The current World State Register $\mathbf{s}_t$.
3. Optional control signals (trajectories, goals, actions).

The denoising process for agent $a$ at generation step $t$ is:

$$x^{(a)}_{t} \sim p_\theta\!\left(x \mid x^{(a)}_{t-W:t-1}, \mathbf{s}_t, c^{(a)}\right)$$

where $c^{(a)}$ is agent-specific conditioning (e.g., ego vehicle pose, camera intrinsics).

### Register Update Rule
After all agents generate step $t$, the register is updated:

$$\mathbf{s}_{t+1} = \text{Attn}(Q = \mathbf{s}_t,\; K = V = [\mathbf{s}_t;\, \mathbf{e}^{(1)}_t;\, \ldots;\, \mathbf{e}^{(A)}_t])$$

where $\mathbf{e}^{(a)}_t = \phi_\text{enc}(x^{(a)}_t)$ is the encoded latent of agent $a$'s newly generated frame. This write-then-broadcast cycle ensures that all agents' observations of the world at step $t$ are consolidated before influencing step $t+1$.

## Technical Formulation

The joint generative distribution factorizes as:

$$p\!\left(x^{(1:A)}_{1:T}\right) = \prod_{t=1}^T p\!\left(\mathbf{s}_t \mid x^{(1:A)}_{<t}, \mathbf{s}_{t-1}\right) \prod_{a=1}^A p_\theta\!\left(x^{(a)}_t \mid x^{(a)}_{t-W:t-1}, \mathbf{s}_t\right)$$

This factorization makes the register a sufficient statistic for cross-agent coordination: each agent only needs the register (not other agents' raw frames) to maintain global consistency. The register compression loss:

$$\mathcal{L}_\text{reg} = \mathbb{E}\!\left[\left\| \mathbf{s}_t - \phi_\text{enc}\!\left(x^{(1:A)}_t\right) \right\|^2\right]$$

is added to the diffusion objective during training to ensure the register faithfully captures world state.

## Experimental Findings

Experiments are conducted on multi-agent autonomous driving simulation using the nuScenes-derived MultiAgent benchmark (6 surround cameras, multiple traffic agent viewpoints):

| Method | FVD ↓ | Cross-Agent Consistency ↑ | Long-Horizon Stability (100 steps) ↑ |
|---|---|---|---|
| Independent per-agent diffusion | 412 | 0.61 | 0.43 |
| Shared history concatenation | 387 | 0.68 | 0.51 |
| **SMAD (World State Register)** | **294** | **0.82** | **0.74** |

FVD = Fréchet Video Distance; lower is better. Consistency and stability metrics are normalized scores in [0, 1].

## Ablation Findings

- **WSR write frequency:** Updating the register every step vs. every 4 steps degrades FVD from 294 to 361, confirming that frequent synchronization is necessary.
- **Register size M:** Performance plateaus at $M = 64$ vectors; smaller registers lose world state fidelity, larger ones add overhead without gain.
- **No local window (global history):** Replacing the local window with full history increases memory quadratically and improves FVD only marginally (288 vs. 294), confirming the local window is an efficient approximation.

## Significance

The World State Register provides a principled, scalable solution to multi-agent video generation consistency. By fixing the cross-agent context to a constant-size shared state, the framework scales to many agents and long rollouts with no increase in per-agent memory cost—a property essential for real-world autonomous driving simulation pipelines.
