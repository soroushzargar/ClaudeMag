# H²MARL: Hierarchical Heterogeneous Multi-agent Reinforcement Learning for Humanoid Robot Control

**Authors:** Xuemin Hu, Kaixin Li, Qingqing Xia, Yuxin Wang, Wei Xie, Bo Tang, Long Chen
**arXiv:** 2611.06523 (cs.RO, cs.MA)
**Submitted:** 4 November 2026
**Venue:** Preprint

---

## Headline

By treating each joint degree of freedom in a humanoid robot as an independent RL agent with explicit physical descriptors, H²MARL achieves coordinated whole-body control that outperforms monolithic controllers and coarse-grouped multi-agent baselines.

---

## Key Findings

- **H²MARL** treats each torque-actuated joint degree of freedom (JDoF) as an independent agent, conditioned on a physical descriptor encoding actuation, kinematic, and dynamic properties.
- A **hierarchical controller** couples a high-level locomotion planner with a low-level executor, trained jointly rather than in isolation.
- A **centralized critic** enables heterogeneous actors to specialize in joint-local behaviors while maintaining global whole-body coordination.
- The method addresses the core failure mode of prior approaches: separate optimization of planning and execution leads to uncoordinated motion when the learned sub-policies have different timescales and state dependencies.

---

## Motivation and Problem

Whole-body humanoid control is among the hardest problems in robot learning. A humanoid's high-dimensional action space spans torque-actuated joints with fundamentally different physical properties: a knee joint and a wrist joint differ in mass, range of motion, speed, and torque capacity. Generic multi-agent algorithms that ignore these differences learn poorly-coordinated behavior; hierarchical controllers that optimize planning and execution separately suffer from interface mismatch when the high-level policy generates plans that the low-level executor cannot physically achieve.

The core question H²MARL addresses: how should a multi-agent RL framework be structured so that joint heterogeneity is explicitly modeled, and the hierarchical coordination between locomotion planning and motor execution is learned jointly?

---

## Method: H²MARL

### JDoF Agents with Physical Descriptors

Each torque-actuated joint degree of freedom is treated as an individual RL agent $i$ with observation $o_i$ augmented by a physical descriptor $d_i$:

$$d_i = [\text{max\_torque}_i,\; \text{range\_of\_motion}_i,\; \text{mass}_i,\; \text{inertia}_i]$$

Each agent outputs a torque command $a_i$ for its joint. The joint-level policy is:

$$\pi_i(a_i \mid o_i, d_i)$$

This fine-grained decomposition allows each agent to specialize its behavior to the physical constraints of its joint without requiring manual grouping or shared architectures across heterogeneous joints.

### Hierarchical Planner-Executor Coupling

A high-level policy $\pi_H$ generates a locomotion plan (target body velocity and orientation) at a coarse timestep. Each JDoF agent (the low-level executor) receives this plan as additional conditioning:

$$\pi_i(a_i \mid o_i, d_i, z_H)$$

where $z_H$ is the plan embedding from $\pi_H$. Both $\pi_H$ and $\{\pi_i\}$ are trained jointly via PPO, with shared rollouts.

### Centralized Critic for Heterogeneous Coordination

Training uses a centralized critic $V(s, \{a_i\}, z_H)$ that observes the full joint state and all actions. The critic provides a shared value estimate that incentivizes actors to coordinate, even though each actor only conditions on its own observation and the plan. This follows the centralized-training, decentralized-execution (CTDE) paradigm, extended here to hierarchical heterogeneous agents.

---

## Experimental Setup and Results

H²MARL is evaluated on whole-body humanoid locomotion tasks requiring simultaneous balance, navigation, and manipulation. Baselines include monolithic controllers, coarsely-grouped multi-agent baselines (grouping all leg joints and all arm joints), and non-hierarchical heterogeneous-agent algorithms.

Key results:
- H²MARL outperforms all baselines on locomotion stability and task success metrics.
- The hierarchical joint training eliminates the motion artifacts (jitter, delayed recovery) observed when planning and execution are trained separately.
- Ablations show that both the physical descriptor conditioning and the centralized critic independently contribute to performance; removing either degrades coordination quality.

---

## Significance

H²MARL proposes a principled decomposition of humanoid control that respects the physical heterogeneity of joints without sacrificing coordination. The joint-as-agent abstraction is elegant: it scales gracefully with the number of degrees of freedom and supports embodiment transfer by swapping physical descriptors rather than retraining policies. The hierarchical joint training eliminates the fundamental interface mismatch that plagues staged training approaches.
