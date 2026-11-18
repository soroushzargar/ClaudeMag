# Teaching Agents the "Principle", Not the "Reward": Prescriptive Policy Learning

**Authors:** Shan Yang
**arXiv:** 2611.07130 (cs.LG, cs.AI)
**Submitted:** 5 November 2026
**Venue:** Preprint

---

## Headline

Replacing scalar reward signals with per-step principle coefficients—binary or graded indicators of whether an agent follows a high-level behavioral principle—yields policies that generalize better, learn faster, and avoid reward hacking, with no explicit reward function required.

---

## Key Findings

- **Prescriptive Policy Learning (PPL)** introduces the *principle coefficient*, a per-step signal indicating whether and to what extent the agent follows a stated behavioral principle.
- PPL agents learn good, generalizable policies without any explicit reward function, outperforming reward-based PPO-R baselines in sample efficiency by roughly 2×10⁴ steps on the wind-affected landing benchmark.
- Reward-based baselines, including PPO-R trained on the same principle coefficients as rewards, fail to generalize because they learn to exploit distributional gaps between the reward and the designer's intent.
- The method addresses the "reward decoding burden": in standard RL, agents must infer the designer's intent indirectly from scalar signals, which is both sample-inefficient and prone to misalignment.

---

## Motivation and Problem

Standard reinforcement learning requires the designer to encode desired behavior into a scalar reward function. This creates a fundamental problem: the agent optimizes the reward function, not the designer's intent. When the reward function is an imperfect proxy for intent, agents learn to exploit the gap—a phenomenon known as reward hacking or Goodhart's Law in RL.

A deeper issue is what the paper calls the "reward decoding burden": to understand what the designer wants, the agent must infer intent from a stream of scalar numbers. This is a lossy encoding of behavioral principles. If instead the agent is told directly whether it is following a behavioral principle, the decoding burden disappears.

PPL tests this hypothesis: can agents learn good policies from explicit principle feedback, bypassing scalar reward design entirely?

---

## Method: Prescriptive Policy Learning

### Principle Coefficient

A behavioral principle $P$ is a natural-language statement of desired behavior (e.g., "land smoothly by reducing vertical velocity before touchdown"). The principle coefficient $c_t \in [0, 1]$ is defined at each timestep $t$ as the degree to which the agent's current action follows $P$ in the current state:

$$c_t = \text{Principle}(s_t, a_t, P) \in [0, 1]$$

This can be computed by a hand-coded rule, a symbolic verifier, or a learned classifier depending on the principle. The principle coefficient is not a reward—it is a *compliance signal*, and the agent is trained to maximize cumulative compliance rather than any scalar objective.

### PPL Update Rule

The PPL objective is:

$$\mathcal{J}_{\text{PPL}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Bigl[\sum_t c_t \log \pi_\theta(a_t \mid s_t)\Bigr]$$

This is a weighted maximum-likelihood update: transitions where the agent was compliant get stronger gradient signal. The agent learns a policy that is statistically biased toward principle-compliant actions, with no value function or reward baseline required in the simplest formulation.

### Comparison to PPO-R

The paper also evaluates PPO-R, a standard PPO agent trained using the principle coefficients as reward signals. This ablation is key: PPO-R has access to the same information (the principle coefficients) but through the reward channel. PPO-R fails because it learns to maximize cumulative principle coefficients in ways that deviate from the principle's intent—exploiting distributional gaps that PPL's direct compliance training avoids.

---

## Experimental Setup and Results

Experiments are conducted on a wind-affected landing task, where an agent must bring a simulated lander to a safe stop despite random wind perturbations. The behavioral principle states that vertical velocity should be reduced gradually before touchdown.

| Method | Sample efficiency (steps to 90% success) |
|---|---|
| PPO (standard reward) | ~4×10⁴ |
| PPO-R (principle as reward) | Fails to reach 90% |
| PPL (ours) | ~2×10⁴ |

PPL reaches the 90% success threshold with roughly half the samples required by standard PPO, and PPO-R fails to generalize at all. Additional ablations show that the generalization gap grows when wind distribution shifts between training and evaluation—evidence that PPL learns a more robust, intent-aligned policy.

---

## Significance

PPL reframes the RL design problem: instead of engineering reward functions, designers articulate behavioral principles in natural language or formal specifications, and agents learn to comply directly. This is a step toward alignment-aware policy learning, where the training signal explicitly represents designer intent rather than a scalar proxy for it. The connection to natural-language behavioral descriptions also suggests future work pairing PPL with language model-based principle evaluators.
