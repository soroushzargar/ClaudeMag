# Training Reasoning Models in Simulation to Natively Use Real Robots

**Authors:** MIT Han Lab (PEAR team)
**arXiv:** 2611.18562 (cs.RO, cs.LG)
**Submitted:** 13 November 2026
**Venue:** Preprint

---

## Headline

A 27B open-weight reasoning model trained entirely in simulation via reinforcement learning—with no expert demonstrations—transfers zero-shot to a real dual-arm robot, raising LIBERO task success from 1.6% to 74.4% and achieving 60% on physical hardware.

---

## Key Findings

- **PEAR** (Post-training Engine for Agentic Robotics) performs robot-use reinforcement learning (RURL) entirely in simulation, requiring no teleoperated demonstrations.
- Starting from a 27B reasoning model (Qwen3.8-27B), RURL on LIBERO improves success from 1.6% to 74.4% in 225 training steps.
- The same checkpoint transfers zero-shot to a real YAM dual-arm robot: 18/30 trials succeed (60%), versus 0% for the base model.
- On RoboLab-120, LIBERO-only training raises success from 5.4% to 22.1%; combined LIBERO+RoboDojo training achieves 37.3%, competitive with π0.5 on the public leaderboard.
- PEAR is built on veRL, LMCache, vLLM, and Ray; code is released at github.com/mit-han-lab/pear.

---

## Motivation and Problem

Generalist robot policies have traditionally relied on large-scale teleoperation datasets—a bottleneck that is expensive, embodiment-specific, and difficult to scale. Meanwhile, large reasoning language models have demonstrated strong capabilities in planning and sequential decision-making on text-based tasks, but their direct application to physical manipulation is hampered by the gap between linguistic reasoning and low-level motor control.

The question this paper addresses: can a large reasoning model acquire practical robot-use capabilities purely through reinforcement learning in simulation, without any demonstrations? If the answer is yes, the cost and embodiment barriers of data collection fall away, and the same model that reasons over language can be turned toward physical robot use with minimal overhead.

---

## Method: PEAR and Robot-Use RL

### Robot-Use RL (RURL)

RURL casts robot manipulation as a language-conditioned tool-use problem. The reasoning model receives a task description, an observation (image + proprioception encoded as tokens), and must output a sequence of motor commands. The environment provides a binary success signal at episode completion.

The RL objective is:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\sum_{t=0}^{T} \gamma^t r_t\bigr]$$

where $r_t = 1$ only at task completion and $r_t = 0$ otherwise. No dense reward shaping or demonstrations are used; the model learns from verifiable binary outcomes.

### PEAR Infrastructure

PEAR orchestrates training across three components:
- **Rollout engine:** vLLM + Ray for distributed episode sampling in the LIBERO simulator.
- **Training engine:** veRL for PPO-based policy gradient updates on the 27B backbone.
- **KV-cache:** LMCache for efficient reuse of observation encodings across timesteps.

The two-stage training schedule first trains on LIBERO for 225 steps, then extends to RoboDojo for generalization.

---

## Experimental Setup and Results

Training was conducted in the LIBERO suite (simulation), and the best checkpoint was evaluated on a real YAM dual-arm robot without any fine-tuning or adaptation.

**Simulation results (LIBERO):**
| Model | LIBERO Success |
|---|---|
| Base Qwen3.8-27B | 1.6% |
| PEAR (LIBERO only) | 74.4% |
| PEAR (LIBERO + RoboDojo) | — |

**Real robot results (YAM dual-arm):**
- Base model: 0/30 trials
- PEAR checkpoint: 18/30 trials (60%)

**RoboLab-120 leaderboard:**
- LIBERO-only checkpoint: 22.1% (+16.7pp over base)
- LIBERO+RoboDojo checkpoint: 37.3% (competitive with π0.5)

---

## Significance

PEAR demonstrates that the data bottleneck of robot learning—large-scale demonstration collection—can be bypassed when a sufficiently capable reasoning model is paired with scalable simulation-based RL. The zero-shot sim-to-real transfer is particularly striking, as the model never observes real robot data during training. This positions reasoning models as a new substrate for robot policy learning, alongside dedicated VLA architectures.
