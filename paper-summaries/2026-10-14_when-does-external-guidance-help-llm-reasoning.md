# When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO

**arXiv:** 2610.06861  
**Venue:** arXiv preprint (October 2026)  
**Authors:** Sofia Torres, Gabriel Almeida, Carter Adams, Camila Rocha  
**Date:** October 2026

---

## One-Line Summary

Guidance-Augmented GRPO (GA-GRPO) provides the first convergence-rate analysis for adding external teacher signals to LLM reinforcement learning: a tight O(1/√T) rate with bias bounded by the total-variation divergence between guided and on-policy distributions, plus a minimax lower bound showing this bias is unavoidable, and an MSE-optimal guidance weight that cuts GPU-hours by 31%.

---

## Problem

Reinforcement learning from verifiable rewards (RLVR) has become the dominant post-training strategy for reasoning LLMs. Methods such as GRPO, PPO, and REINFORCE generate rollouts on-policy and update the model based on correctness signals. Recently, guidance-augmented variants add external signals—teacher chain-of-thought demonstrations, search-engine results, or intermediate verification checks—to the rollout distribution, reporting empirical gains.

However, **no theoretical analysis explains when guidance helps or hurts**, what the optimal amount of guidance is, or how the bias introduced by off-policy guidance interacts with convergence. Practitioners are left to tune guidance weights by grid search, which is expensive and produces unstable results.

---

## Key Contribution: Bias-Variance Decomposition of Guidance

The paper frames guidance as an operator that shifts the sampling distribution from the on-policy $p_\theta$ to a guided distribution $q$. The core insight is:

$$\text{Error}(\text{GA-GRPO}) = \underbrace{\text{Variance}(\hat{g}_t)}_{\text{finite-sample}} + \underbrace{\text{Bias}^2(q, p_\theta)}_{\text{guidance bias}}$$

The **variance term** decreases with more rollouts (standard GRPO effect). The **bias term** is fixed by the quality of the external guidance signal and is bounded above by:

$$\text{Bias}(q, p_\theta) \leq C \cdot \text{TV}(q, p_\theta)$$

where $\text{TV}(q, p_\theta)$ is the total-variation divergence between the guided and on-policy distributions, and $C$ is a smoothness constant.

---

## Why Existing Approaches Fall Short

- **GRPO / REINFORCE:** Purely on-policy; variance is high for sparse rewards but bias is zero.
- **SFT from demonstrations:** Zero variance on the demonstration distribution but high bias—the model drifts toward the teacher and away from its own competence.
- **Ad-hoc guidance weighting (TAPO, LUFFY, ExPO):** Report gains empirically but provide no principled weighting rule; different papers use guidance weights ranging from 0.01 to 10.
- **DPO / offline RL:** Off-policy by construction; high bias with no principled debiasing mechanism.

---

## Core Method: Guidance-Augmented GRPO with Optimal Weighting

**Guidance-Augmented GRPO (GA-GRPO):**

Let $\theta_t$ be the current policy. At each step $t$, GA-GRPO draws rollouts from a mixture:

$$\tilde{q}_t = (1 - \alpha) \cdot p_{\theta_t} + \alpha \cdot q$$

where $q$ is the guidance distribution (e.g., a teacher model) and $\alpha \in [0, 1]$ is the guidance weight. The policy gradient estimator uses importance weights to correct for the off-policy sampling:

$$\hat{g}_t = \frac{1}{K}\sum_{k=1}^{K} \frac{p_{\theta_t}(\tau_k)}{\tilde{q}_t(\tau_k)} \cdot r(\tau_k) \cdot \nabla_\theta \log p_\theta(\tau_k)$$

**Optimal guidance weight:** The MSE of $\hat{g}_t$ is:

$$\text{MSE}(\alpha) = \frac{\sigma^2(\theta_t)}{K(1-\alpha)} + \alpha^2 \cdot C^2 \cdot \text{TV}^2(q, p_{\theta_t})$$

Minimizing over $\alpha$ gives the closed-form optimal:

$$\alpha^* = \frac{\sigma^2(\theta_t) / K}{\sigma^2(\theta_t)/K + 2 C^2 \cdot \text{TV}^2(q, p_{\theta_t})}$$

This is the paper's central result: use more guidance when variance is high (early training, sparse rewards) and less when bias is high (teacher far from the current policy).

---

## Technical Formulation

Let $\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q}[r(\tau) \log p_\theta(\tau)]$. The convergence theorem states:

**Theorem (informal).** Under $L$-smooth policy gradients and bounded guidance divergence $\text{TV}(q, p_\theta) \leq \delta$, GA-GRPO with step size $\eta_t = O(1/\sqrt{t})$ satisfies:

$$\frac{1}{T}\sum_{t=1}^{T} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \frac{C_1}{\sqrt{T}} + \delta^2 C_2$$

where $C_1, C_2$ depend on $L$, the reward range, and $K$. The first term is the standard GRPO rate; the second is the unavoidable guidance bias.

**Minimax lower bound.** For any guidance-augmented algorithm, there exists a problem instance with $\text{TV}(q, p_\theta) \geq \delta$ such that:

$$\frac{1}{T}\sum_{t=1}^T \|\nabla \mathcal{L}(\theta_t)\|^2 \geq \Omega\!\left(\frac{1}{\sqrt{T}}\right) + \Omega(\delta^2)$$

This shows the $\delta^2$ bias term cannot be eliminated by any algorithm that uses guidance with fixed divergence $\delta$.

---

## Learning or Inference Procedure

1. Specify a teacher distribution $q$ (e.g., GPT-4o chain-of-thought traces on the training problems).
2. Estimate $\text{TV}(q, p_\theta)$ on a calibration set by computing KL divergence and using the Pinsker inequality.
3. Compute $\alpha^*$ using the closed-form formula with $\sigma^2$ estimated from on-policy rollout variance.
4. Run GA-GRPO with the mixture $\tilde{q}_t = (1-\alpha^*) p_{\theta_t} + \alpha^* q$, updating $\alpha^*$ every 100 gradient steps as $\theta$ changes.
5. Evaluate on held-out benchmarks.

---

## What the Paper Claims

- GA-GRPO with optimal $\alpha^*$ matches or surpasses TAPO, LUFFY, ExPO, and vanilla GRPO on all nine benchmarks.
- The optimal $\alpha^*$ is interpretable: it is small early in training (teacher is far, guidance is biased), grows as the model improves toward the teacher, then shrinks again as variance drops.
- 31% fewer GPU-hours than vanilla GRPO at matched final accuracy on MATH-500.

---

## Experimental Findings

- **Model:** Qwen2.5-Math-7B-Base fine-tuned with GA-GRPO.
- **Teacher:** GPT-4o chain-of-thought traces (50K examples).
- **Benchmarks:** MATH-500, AIME 2024/2025, AMC 2023, GPQA-Diamond, OlympiadBench, Minerva, Olympus, GaoKao.
- **Key results:** GA-GRPO with $\alpha^*$ achieves 82.4% on MATH-500 vs. 80.1% for vanilla GRPO.
- **Efficiency:** 3.2M GPU-minutes for GA-GRPO vs. 4.6M for vanilla GRPO at matched accuracy.
- **Comparison:** Surpasses TAPO (+1.1%), LUFFY (+0.7%), ExPO (+2.3%) averaged across benchmarks.

---

## Ablations and Interpretation

- **Fixed $\alpha$ vs. $\alpha^*$:** Fixed $\alpha = 0.3$ (the most common choice in prior work) achieves 80.8% on MATH-500; $\alpha^* = 0.3$ achieves 82.4%, suggesting the adaptive scheduling is the key gain.
- **TV estimation method:** KL-based estimation is within 0.5% of exact TV on held-out calibration sets; cheaper entropy-based proxies are noisier but achieve 80.2% accuracy.
- **Teacher quality:** When $\text{TV}(q, p_\theta)$ is large (low-quality teacher), $\alpha^*$ correctly drives toward zero, recovering vanilla GRPO; the method is teacher-agnostic.
- **$\delta^2$ bias confirmation:** Models trained with a highly off-distribution teacher ($\delta = 0.8$) achieve lower accuracy despite a large guidance signal, confirming the minimax lower bound.

---

## Reference

Sofia Torres, Gabriel Almeida, Carter Adams, Camila Rocha. **When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO**. arXiv:2610.06861.  
https://arxiv.org/abs/2610.06861
