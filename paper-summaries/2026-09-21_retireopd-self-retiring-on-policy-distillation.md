# RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning

**arXiv:** 2609.20784  
**Submitted:** September 17, 2026  
**Authors:** Yan Yu, Zhengxi Lu, Yizhou Liu, Yichen Pan, Aozhe Wang, Qipeng Chen, Hua Yang, Wenqi Zhang, Weiming Lu, Qianglong Chen, Yongliang Shen  
**Affiliation:** Zhejiang University; Alibaba Group  
**Venue:** arXiv preprint

---

## Headline Finding

RetireOPD improves multi-turn agentic RL training by first optimizing a skill-conditioned teacher, then training a skill-free student jointly with RL and on-policy distillation (OPD), and finally letting the student automatically retire the teacher — dropping distillation — once the student's improvement stagnates and it reaches a target success fraction; this adaptive retirement achieves +14.1–18.8% success rate on ALFWorld and +11.8–19.0% on WebShop over pure RL baselines across three model scales.

---

## Key Findings (Pyramid Layer 2)

1. **Sparse rewards in agentic RL motivate distillation.** Multi-turn agents receive a single binary reward at trajectory end; this sparse signal provides weak learning signal for individual token choices. On-policy distillation (OPD) addresses this by supplying dense token-level supervision from a teacher that has privileged access to task-relevant skills (e.g., a ground-truth action plan or additional environment state).

2. **Privileged information does not guarantee teacher reliability.** A teacher conditioned on privileged task skills can still produce incorrect reasoning in agentic domains where perception and world-state uncertainty are high. Simply forcing the student to imitate an unreliable teacher introduces harmful supervision that degrades performance, particularly in early training when neither teacher nor student is well-calibrated.

3. **Distillation benefit is stage-dependent.** In early training, the teacher is systematically better than the student and distillation provides a reliable learning signal. In later training, as the student approaches the teacher's competence, continued OPD can cause the student to regress toward the teacher's suboptimal behaviors rather than explore further improvements. A static distillation schedule cannot adapt to this dynamic.

4. **Adaptive Retirement solves both problems.** RetireOPD monitors the KL divergence between student and teacher trajectories (the discrepancy) and the student's success rate relative to the teacher's. When the discrepancy stops decreasing (student is no longer learning from the teacher) AND the student has reached a target fraction $\alpha$ of the teacher's success rate, distillation is permanently retired and RL continues alone. This data-driven switch point is learned rather than scheduled.

---

## Methodology (Pyramid Layer 3)

### Background: On-Policy Distillation

Standard OPD for LLM agents defines a teacher $\pi_T(a_t \mid s_t, z)$ conditioned on privilege information $z$ (e.g., the complete task plan) and a student $\pi_S(a_t \mid s_t)$ without privilege. The student is trained with:
$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_\tau \sum_t \text{KL}\!\left(\pi_T(a_t \mid s_t, z) \,\|\, \pi_S(a_t \mid s_t)\right)$$

The student is simultaneously trained with a RL objective (e.g., GRPO) on the sparse environment reward:
$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_\tau\!\left[G_\tau \log \pi_S(a_t \mid s_t)\right]$$

with $G_\tau$ the trajectory return. The joint loss is $\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta \mathcal{L}_{\text{OPD}}$.

### RetireOPD: Decoupled Teacher Training

The key departure from prior OPD is that RetireOPD trains the teacher separately before student training begins:

1. **Teacher pre-training:** Initialize teacher from the base LLM. Fine-tune with GRPO on the environment reward augmented with a skill-following reward that encourages the teacher to use the provided privilege $z$:
$$r_T = r_{\text{env}} + \lambda_z \cdot \mathbf{1}[\pi_T \text{ uses } z]$$

2. **Student joint training:** Initialize student from the same base LLM. Train jointly:
$$\mathcal{L}_S = \mathcal{L}_{\text{RL}}(\pi_S) + \beta \cdot \mathcal{L}_{\text{OPD}}(\pi_T, \pi_S)$$

The teacher parameters are frozen during student training.

### Adaptive Retirement Criterion

At training step $t$, compute:
- **Discrepancy:** $\Delta_t = \text{KL}(\pi_T \| \pi_S)$ averaged over a recent batch of student rollouts.
- **Success gap:** $\rho_t = R_S^t / R_T$, where $R_S^t$ is the student's rolling success rate and $R_T$ is the teacher's fixed evaluation success rate.

The retirement condition is:
$$\text{Retire if: } \Delta_t - \Delta_{t-k} < \epsilon_{\Delta} \text{ AND } \rho_t \geq \alpha$$

where $k$ is a rolling window (default: 100 steps), $\epsilon_\Delta$ is a discrepancy-convergence threshold, and $\alpha \in (0, 1)$ is the target success fraction (default: 0.85). After retirement, $\beta$ is set to 0 and only $\mathcal{L}_{\text{RL}}$ is optimized.

---

## Technical Formulation

The full RetireOPD training objective before retirement:
$$\mathcal{L}_{\text{RetireOPD}} = \mathcal{L}_{\text{GRPO}}(\pi_S) + \beta \cdot \mathbb{E}_\tau \left[\sum_t \text{KL}\!\left(\pi_T(\cdot \mid s_t, z) \,\|\, \pi_S(\cdot \mid s_t)\right)\right]$$

where $\mathcal{L}_{\text{GRPO}}$ is the group-relative policy optimization loss:
$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_\tau \sum_t \min\!\left(\frac{\pi_S(a_t)}{\pi_{\text{ref}}(a_t)} \hat{A}_t,\, \text{clip}\!\left(\frac{\pi_S}{\pi_{\text{ref}}}, 1\pm\epsilon\right)\hat{A}_t\right)$$

After retirement, only $\mathcal{L}_{\text{GRPO}}$ is optimized, with the student acting as its own reference model from the retirement checkpoint.

---

## Learning or Inference Procedure

1. Pre-train the skill-conditioned teacher with GRPO + skill reward for $T_{\text{teacher}}$ steps.
2. Initialize student from base LLM.
3. Jointly train student with $\mathcal{L}_{\text{RetireOPD}}$.
4. At each step, evaluate retirement condition; if triggered, set $\beta=0$ and continue with RL only.
5. At inference, use only the skill-free student $\pi_S$; the teacher and privilege $z$ are not available.

---

## What the Guarantee Says

RetireOPD does not provide a formal convergence guarantee. The paper provides a theoretical motivation: under a simplifying assumption that the teacher's policy is a noisy oracle (correct with probability $p > 0.5$), continued OPD past the student's saturation point has negative expected gradient alignment with the true reward. The retirement condition is designed to detect this saturation empirically before it causes regression.

---

## Experimental Findings

### Models
- Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B (base LLMs)

### Benchmarks
- **ALFWorld:** Text-based household navigation and manipulation (6 task types, 134 test episodes)
- **WebShop:** Web-based product search and purchase (500 test episodes)

### Main Results (7B scale, success rate %)

| Method | ALFWorld | WebShop |
|--------|---------|---------|
| SFT | 54.2 | 59.3 |
| GRPO (RL only) | 67.1 | 68.4 |
| Continuous OPD+RL | 71.3 | 72.1 |
| SAGE-OPD | 73.8 | 74.5 |
| **RetireOPD** | **85.9** | **87.4** |

RetireOPD outperforms continuous joint OPD+RL by 14.6% on ALFWorld and 15.3% on WebShop at the 7B scale.

---

## Ablations and Interpretation

- **No teacher pre-training (shared teacher-student init):** Success drops by 8.3% on ALFWorld; the teacher is unreliable at the start and corrupts early student gradients.
- **Fixed distillation schedule (no retirement):** Success drops by 6.4% on ALFWorld; continued OPD past saturation actively hurts.
- **Retirement based on discrepancy only (ignoring success gap):** Student retires too early at 1B scale before reaching competence, losing 4.1%.
- **Retirement based on success gap only (ignoring discrepancy):** Retirement delayed too long at 7B scale, spending excess steps on unhelpful OPD, losing 2.7%.
- **$\alpha$ sensitivity:** $\alpha \in [0.80, 0.90]$ is robust; $\alpha < 0.70$ causes premature retirement and $\alpha > 0.95$ causes over-extended distillation.

---

## Reference

Yan Yu, Zhengxi Lu, Yizhou Liu, Yichen Pan, Aozhe Wang, Qipeng Chen, Hua Yang, Wenqi Zhang, Weiming Lu, Qianglong Chen, and Yongliang Shen. **RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning.** arXiv:2609.20784, September 2026. https://arxiv.org/abs/2609.20784
