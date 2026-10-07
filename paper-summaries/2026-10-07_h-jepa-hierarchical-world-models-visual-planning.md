# H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning

**arXiv:** 2610.06805  
**Authors:** Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, Randall Balestriero  
**Affiliation:** Meta AI; New York University  
**Date:** October 5, 2026

---

## One-Line Summary

H-JEPA stacks action-conditioned JEPAs hierarchically so that each level learns to predict farther ahead in its own latent space, enabling a 3-level hierarchy to raise Visual AntMaze success from 18% to 73% with less planner compute.

---

## Problem

Long-horizon planning requires reasoning across multiple timescales simultaneously: sub-second muscle-level control, second-scale motion planning, and minute-scale goal-directed navigation. Flat world models trained at a single temporal resolution conflate these timescales, forcing the model to predict fine-grained sensory detail even when planning a long-horizon route—wasting capacity and making effective planning computationally expensive. A principled approach to hierarchical abstraction in latent world models remains an open challenge.

---

## Why Existing Approaches Fall Short

- **Flat world models are computationally wasteful:** A single-level world model predicts every future state at the same resolution, forcing the planner to reason about irrelevant low-level detail when making high-level decisions.
- **Hand-designed hierarchies:** Prior hierarchical RL methods (options frameworks, HIRO, HAC) pre-specify the timescale hierarchy and require human-designed sub-goals, lacking end-to-end trainability.
- **Generative hierarchical models are fragile:** Generative world models that try to predict raw pixels hierarchically accumulate errors across levels and are sensitive to the choice of pixel-prediction loss.
- **JEPA-style models predict in latent space but remain flat:** Existing V-JEPA and related work learn single-level latent predictors; extending them to multiple timescales in an end-to-end trainable manner is non-trivial.

---

## Core Method

**H-JEPA (Hierarchical Joint Embedding Predictive Architecture)** stacks $L$ levels of action-conditioned JEPAs:

- **Level 1** operates at the native frame rate, predicting the latent representation of the next frame from the current latent and action.
- **Level $\ell > 1$** operates at a $k$-times-coarser timescale. It receives the averaged (temporally pooled) latent from level $\ell-1$ and predicts the level-$(\ell-1)$ latent $k$ frames ahead.
- **End-to-end training:** All levels are trained jointly with a shared JEPA objective: minimize the distance between the predicted latent and the target (stop-gradient) latent from the lower level, using a VICReg-style regularizer to prevent representational collapse.

The key property that emerges from this hierarchy is **timescale separation**: when the world has factors evolving at different speeds (e.g., agent position changes fast; task-relevant object state changes slowly), higher levels learn to discard fast, unpredictable detail and retain the slow, task-relevant state—automatically without any explicit supervision.

---

## Technical Formulation

Let $z_t^\ell$ denote the latent at level $\ell$ and time step $t$. Define temporal pooling $P_k$ that averages $k$ consecutive latents. The H-JEPA prediction target at level $\ell$ is:
$$z_t^{\ell} = P_k(z_{t \cdot k}^{\ell-1}, \ldots, z_{(t+1) \cdot k - 1}^{\ell-1})$$

The predictor $f^\ell_\phi$ at level $\ell$ predicts $k$ steps ahead:
$$\hat{z}_{t+1}^\ell = f^\ell_\phi(z_t^\ell, a_{t \cdot k : (t+1) \cdot k})$$

where $a_{t \cdot k : (t+1) \cdot k}$ is the sequence of actions during the predicted interval.

The per-level JEPA loss is:
$$\mathcal{L}^\ell = \| \hat{z}_{t+1}^\ell - \text{sg}(z_{t+1}^\ell) \|^2 + \lambda \cdot \mathcal{L}_\text{VICReg}(z^\ell)$$

The total objective sums over all levels:
$$\mathcal{L}_\text{H-JEPA} = \sum_{\ell=1}^{L} \mathcal{L}^\ell$$

**Planning** uses a latent MCTS or MPPI planner applied at the highest level $L$, with each evaluation of the rollout scoring function computed in the level-$L$ latent space. This dramatically reduces the compute per planning node relative to planning at level 1.

---

## Learning or Inference Procedure

**Training:**
1. Collect trajectories $(o_1, a_1, \ldots, o_T, a_T)$ using a behavior policy.
2. Encode observations with a shared encoder $g_\psi$ to obtain $z^0 = g_\psi(o)$.
3. Pool and encode each level: $z^1 = P_k(z^0)$, $z^2 = P_k(z^1)$, etc.
4. Train all levels jointly end-to-end with $\mathcal{L}_\text{H-JEPA}$.
5. Optionally, add inverse-dynamics heads at each level for action prediction from consecutive latents.

**Planning at test time:**
1. Encode the current observation to $z^0$; propagate to level $L$.
2. Run a planner (MPPI or MCTS) in the level-$L$ latent space.
3. Execute the first action of the best plan; receive next observation; repeat.

---

## What the Guarantee Says

No formal convergence guarantee is provided. The central empirical claim is: **timescale separation emerges automatically from end-to-end training** when the data contains factors evolving at different speeds. This is validated by probing level-$\ell$ representations for factors of variation at different timescales. Additionally, the paper provides ablations confirming that the hierarchy is necessary—a flat JEPA with the same total parameter count does not match the hierarchical planner's performance.

---

## Experimental Findings

- **Visual AntMaze:** A three-level H-JEPA hierarchy raises success rate from **18% (flat JEPA) to 73%** while using less planner compute.
- **Manipulation environments:** H-JEPA outperforms flat JEPA on robotic pick-and-place and peg-insertion tasks requiring multi-step coordination.
- **DROID (real robot):** With inverse-dynamics supervision, H-JEPA improves offline planning fidelity on diverse real-robot demonstration data.
- **Timescale separation:** Analysis confirms that level-2 and level-3 representations encode task-relevant state (object positions, goal proximity) while discarding fast-varying visual detail.

---

## Ablations and Interpretation

**Number of levels:** Performance consistently improves from 1 to 3 levels; 4 levels provides marginal additional gain but adds training complexity.

**Pooling stride $k$:** Smaller strides (k=2) produce a finer hierarchy that helps in tasks with fast-changing task-relevant state; larger strides (k=8) help in tasks with very long horizon requirements.

**Timescale analysis:** Probing each level's latent for low-level visual features vs.\ high-level task state confirms the expected hierarchy: level 1 is dominated by visual texture; level 3 is almost entirely task-state driven.

**Comparison to hand-designed hierarchies:** H-JEPA matches or exceeds options-based and HIRO-style methods on standard benchmarks while requiring no manual specification of sub-goals or timescale boundaries.

---

## Reference

Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, Randall Balestriero.
**H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning.**
arXiv:2610.06805, October 2026.
https://arxiv.org/abs/2610.06805
