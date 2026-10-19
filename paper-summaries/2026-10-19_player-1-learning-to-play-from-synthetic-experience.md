# Player-1: Learning to Play from Synthetic Experience

**arXiv:** 2610.20584  
**Venue:** arXiv preprint, October 2026  
**Authors:** Mingyu Ouyang, Yiran Wang, Siyuan Hu, Xingyilang Yin, Huiqiang Sun, Guangzhi Wang, Junfu Pu, Cheng Cheng, Honghao Chen, Ying Shan, Mike Zheng Shou, Hwee Tou Ng  
**Affiliations:** National University of Singapore (NUS), Tencent ARC, Tencent IEG, Nanyang Technological University (NTU)  
**Date:** October 14, 2026

---

## One-Line Summary

By training a vision-language agent with PPO in a suite of 16 synthetic 2D and 3D game environments, Player-1 triples the MineDojo-300 success rate of the best prior baseline while transferring to held-out games it was never trained on.

---

## Problem

Generalist GUI agents built on vision-language models excel at desktop and browser tasks, but digital gameplay demands a qualitatively different set of motor skills: sustained directional inputs, relative pointer motion across frames, and composite simultaneous key-press-plus-mouse actions. Standard computer-use action spaces represent discrete, stateless clicks and keystrokes; they cannot encode the continuous, feedback-driven control loops that distinguish aiming from clicking, driving from navigating menus, or combat from typing. Without a matching action representation and reward signal, RL fine-tuning cannot improve these capabilities.

---

## Key Observation: Action Space Determines What RL Can Improve

Standard computer-use action spaces omit:
1. **Relative pointer motion** — joystick-like control required for aiming and driving.
2. **Sustained inputs** — held keys and buttons required for running and shooting.
3. **Composite actions** — simultaneous multi-key sequences required for combat combos.

Without these primitives in the action vocabulary, any RL agent trying to learn to play games will fail not because the policy is wrong, but because the policy cannot express the correct action.

---

## Why Existing Approaches Fall Short

- **SFT-only agents:** Can imitate successful play trajectories, but imitation is brittle to novel game states and does not generalize to new games.
- **Standard GUI agents (text-based):** Treat games as web navigation; cannot sustain directional control or execute combo actions.
- **Existing RL game agents:** Require game-specific reward shaping and environment wrappers; they do not generalize across game types or transfer to VLM-based GUI agent architectures.
- **Real-game RL training:** Requires running actual game executables, which is slow, platform-dependent, and legally constrained for commercial games.

---

## Core Method: Player-1

**Extended Action Space:** Player-1 adds relative pointer motion (delta-x, delta-y), sustained-input primitives (hold/release), and composite action macros to the standard computer-use vocabulary.

**SFT Phase:** Fine-tunes on 6,044 curated human gameplay trajectories spanning aiming, combat, driving, navigation, and UI interaction tasks.

**Player-1-Gym:** A suite of 16 synthetic 2D and 3D environments built to cover five control families:
1. Aiming
2. Combat
3. Driving
4. Navigation
5. UI interaction

The environments are procedurally generated so that each RL episode provides a novel layout, preventing simple memorization.

**RL Phase:** PPO is run in Player-1-Gym using game outcome and progress signals as rewards. The VLM policy receives raw pixel observations and must produce extended-action-space actions. Training runs asynchronously, with 9B and 35B-A3B model sizes.

**Evaluation:** Held-out benchmarks include MineDojo-300 (Minecraft task suite), ViZDoom-10 (first-person shooter), and GameWorld—none of which appear in Player-1-Gym.

---

## Technical Formulation

Let $\mathcal{A}_{\text{std}}$ be the standard GUI action space and $\mathcal{A}_{\text{ext}}$ be the extended action space:

$$\mathcal{A}_{\text{ext}} = \mathcal{A}_{\text{std}} \cup \mathcal{A}_{\text{rel}} \cup \mathcal{A}_{\text{hold}} \cup \mathcal{A}_{\text{combo}}$$

where $\mathcal{A}_{\text{rel}}$ is the set of relative-motion actions $(\Delta x, \Delta y) \in \mathbb{R}^2$, $\mathcal{A}_{\text{hold}}$ is the set of hold/release primitives, and $\mathcal{A}_{\text{combo}}$ is the set of composite key combinations.

The PPO objective is:

$$\mathcal{L}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t | o_t) / \pi_{\theta_{\text{old}}}(a_t | o_t)$ is the probability ratio, $\hat{A}_t$ is the generalized advantage estimate, and $o_t$ is the pixel observation at step $t$.

The SFT loss over demonstration trajectories $\mathcal{D} = \{(o_t, a_t)\}$ is standard next-token prediction on the extended vocabulary.

---

## Learning or Inference Procedure

1. **Collect SFT data:** 6,044 human gameplay trajectories annotated with extended-action-space labels.
2. **SFT fine-tune:** Optimize $\mathcal{L}_{\text{SFT}}$ on both base model sizes (9B and 35B-A3B).
3. **Deploy to Player-1-Gym:** Run each environment with a short timeout per episode; reward signals are success/failure and task-progress proxies.
4. **PPO training:** Run asynchronously with rollout workers and a central policy updater; no game-specific reward engineering beyond success signals.
5. **Evaluation:** Zero-shot inference on MineDojo-300, ViZDoom-10, and GameWorld using the RL-trained policy.

---

## What the Paper Claims

- Player-1-9B raises MineDojo-300 success from 12.63% (best prior baseline) to **38.33%**, a +25.7 pp absolute gain.
- Player-1-35B-A3B reaches **41.04%**, a +17.7 pp gain over its own SFT policy.
- Gains transfer to ViZDoom-10 and GameWorld without any game-specific fine-tuning.
- The SFT-only ablation scores substantially below the PPO-trained model, confirming that RL—not just better demonstrations—drives the improvement.

---

## Experimental Findings

- **MineDojo-300:** Player-1-9B: 38.33%; Player-1-35B-A3B: 41.04%; best prior: 12.63%.
- **ViZDoom-10:** Positive transfer, exact numbers not surfaced in available secondary sources.
- **GameWorld:** Positive transfer; model not trained on any ViZDoom or GameWorld environment.
- **SFT ablation:** SFT-only policy substantially lower than PPO policy on all evaluated games.
- **Control family breakdown:** Largest absolute gains on aiming and combat tasks, which require the relative-motion and sustained-input primitives most absent from standard action spaces.

---

## Ablations and Interpretation

- **Action space ablation:** Removing relative-motion actions causes a large drop on aiming and driving tasks; removing composite actions degrades combat performance.
- **Gym environment count:** Performance improves with number of training environments, suggesting diversity is key to generalization.
- **SFT quality:** Removing the SFT warm-start and training RL from the base model significantly slows convergence and reduces final performance.
- **Model scale:** 35B-A3B outperforms 9B on MineDojo-300 (+2.71 pp); the gap is larger on ViZDoom, suggesting game reasoning benefits more from scale than game motor control.
- **Why transfer works:** The five control families in Player-1-Gym cover the motor primitives needed for most PC games, so RL-acquired motor skills generalize even when the visual appearance and reward structure differ.

---

## Reference

Mingyu Ouyang, Yiran Wang, Siyuan Hu, et al. **Player-1: Learning to Play from Synthetic Experience**. arXiv:2610.20584. October 2026.  
https://arxiv.org/abs/2610.20584
