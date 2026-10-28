# TEMPO: Efficient Test-Time Adaptation For Robot Policies

**Authors:** Motasem Alfarra, Pietro Mazzaglia, Markus Peschl, Daniel Dijkman, Christos Louizos  
**Venue:** NeurIPS 2026  
**arXiv:** 2610.20318  
**Date:** October 14, 2026  
**URL:** https://arxiv.org/abs/2610.20318

---

## Summary

Reinforcement-learned robot locomotion policies trained in simulation frequently degrade in deployment when conditions differ from training—payloads shift the center of mass, terrain changes friction, sensors drift with temperature. While the deployed robot's own observations contain enough information to adapt, test-time adaptation of RL policies has been limited by training instability and poor sample efficiency. TEMPO solves this with an entropy regularizer on the critic combined with a score-based memory buffer that upweights hard observations, enabling efficient and stable on-device adaptation with no access to a simulator.

---

## Problem Setting

Given a policy π_θ trained offline (in simulation), the goal is to adapt θ at test time using only the observations gathered during deployment—without a simulator, without ground-truth reward labels, and within a tight compute budget (single device, no cloud). Distribution shifts include: added mass/payload, terrain friction changes, joint actuator degradation, and proprioceptive sensor noise.

---

## Method: TEMPO

### Entropy Regularization on the Critic

The policy and critic are updated jointly during test-time adaptation. The standard RL objective is augmented with an entropy regularizer on the critic's value predictions:

**L_critic = L_TD + β · H(V(s))**

where H(V(s)) penalizes overconfident value estimates. This prevents the critic from collapsing to a degenerate solution under distribution shift, stabilizing the policy gradient updates.

### Score-Based Memory

TEMPO maintains a replay buffer with a score-based prioritization scheme. Each observation's score is an estimate of how "hard" it was for the current policy—high-scoring observations are sampled more frequently. The scores are updated online using the critic's TD errors, requiring no additional labeled data. Hard observations that the policy fails to handle are oversampled, accelerating adaptation where it is most needed.

---

## Key Results

### Simulation Benchmarks (MuJoCo Playground)

Evaluated on three quadruped platforms (Go1, Go2, G1) under four distribution shift types:
- Up to **80% performance improvement** in the single-device setting over the non-adapted baseline
- Consistent gains across all four shift types and all three robot morphologies
- Outperforms standard PPO fine-tuning and test-time normalization baselines

### Physical Unitree Go2 Robot

On-device adaptation with TEMPO on a real Unitree Go2 carrying an added payload (simulating a delivery scenario):
- Recovers baseline performance within 10 minutes of on-device adaptation
- Non-adapted policy degrades significantly; PPO fine-tuning diverges

---

## Significance

TEMPO demonstrates that efficient, stable test-time adaptation of locomotion policies is achievable on real hardware within a practically deployable compute budget. The combination of critic entropy regularization and score-based memory is orthogonal to the choice of base policy and can be applied to any PPO-based locomotion system.

---

## Citation

Motasem Alfarra, Pietro Mazzaglia, Markus Peschl, Daniel Dijkman, Christos Louizos.  
**Efficient Test-Time Adaptation For Robot Policies.**  
arXiv:2610.20318. NeurIPS 2026.  
https://arxiv.org/abs/2610.20318
