# Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO

**arXiv:** 2608.27351  
**Authors:** Yunpeng Ba, Zhi Zheng, Yue Xie, Jiaqing Li, Xialiang Tong, Tao Zhong, Mingxuan Yuan, Zhichao Lu, Xuyang Wu, Zhenkun Wang  
**Submitted:** August 28, 2026  
**Area:** LLM Post-Training, Reasoning, Evolution Strategies

---

## Summary

This paper provides a theoretical and empirical understanding of why Evolution Strategies (ES) — a population-based, gradient-free optimization method — produce broader solution coverage than GRPO and similar RL post-training approaches for LLM reasoning. The key insight is that RL methods narrow the output distribution around high-reward solutions, collapsing pass@k diversity, while ES optimizes in weight space and naturally maintains distributional breadth.

## Problem

LLM reasoning post-training methods like GRPO optimize the model's expected reward on a given prompt by updating policy parameters based on sampled outputs and their rewards. While this raises average accuracy (pass@1), it narrows the model's output distribution: the model learns to produce a small number of high-reward output patterns, reducing the diversity of solutions it generates across multiple samples.

This matters for **pass@k** metrics — the probability that at least one of $k$ sampled outputs is correct — which are central to scientific discovery, mathematical exploration, and code generation settings where diverse candidate solutions are valuable. A model that achieves high pass@1 by collapsing to a single strategy may actually degrade in pass@k relative to a less fine-tuned model.

## Method

**Evolution Strategies (ES)** treats the model as a black box and optimizes in weight space rather than in policy output space. At each iteration:

1. Sample a population of weight perturbations $\{\epsilon_i\}_{i=1}^N$ from a standard normal distribution
2. Evaluate each perturbed model $\theta + \sigma \epsilon_i$ on the current batch of problems, collecting rewards $\{R_i\}$
3. Update: $\theta \leftarrow \theta + \frac{\alpha}{N\sigma} \sum_i R_i \epsilon_i$

Because ES perturbs weights globally rather than conditioning on specific prompts, the model is never forced to commit to a specific high-reward output pattern. The weight updates spread probability mass across diverse weight configurations, maintaining output diversity.

**Why ES avoids distribution collapse:** The paper formalizes this as follows. GRPO updates increase the logit gap between selected and rejected outputs conditioned on a prompt $x$. Over many iterations, this mechanically reduces $H(\pi_\theta(\cdot|x))$ (output entropy). ES updates instead optimize $\mathbb{E}[\sum_i R_i \epsilon_i]$ over weight perturbations and have no direct dependence on per-prompt output distributions, leaving output entropy regulated by the landscape of weight perturbations rather than forced to collapse.

## Technical Formulation

The ES update applied to parameters $\theta$ is:
$$\theta_{t+1} = \theta_t + \frac{\alpha}{N\sigma}\sum_{i=1}^N R(\theta_t + \sigma\epsilon_i)\,\epsilon_i$$

where $\alpha$ is the learning rate, $\sigma$ is the perturbation scale, $N$ is the population size, and $R(\theta')$ is the total reward of model $\theta'$ on the current problem batch.

The paper proves that, under mild regularity conditions, the ES update is an unbiased estimate of $\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[R(\theta + \sigma\epsilon)]$, i.e., the gradient of the smoothed objective. Crucially, this smoothed objective averages over a neighborhood of weights rather than a single $\theta$, so maximizing it does not require concentrating probability mass at a single output pattern.

The contrast with GRPO: GRPO's update is $\nabla_\theta \log \pi_\theta(a|x) \hat{A}$ averaged over rollouts, which directly increases the log-probability of high-advantage outputs, accelerating convergence at the cost of diversity.

## Experimental Results

Evaluated on math reasoning (MATH-500, AMC/AIME) and code synthesis (LiveCodeBench) benchmarks using Qwen2.5-7B as the base model:

- **pass@1:** ES matches GRPO within 1–2 percentage points across all benchmarks
- **pass@8:** ES outperforms GRPO by +6.3 pp on MATH-500, +11.4 pp on AMC/AIME, +8.7 pp on LiveCodeBench
- **pass@32:** ES advantages grow further, with +14.2 pp on AMC/AIME
- **Solution diversity:** ES-trained models produce measurably more varied solution strategies; unique solution fingerprints (distinct reasoning paths) are 2.3× more frequent under ES than GRPO at $k=8$
- **Output entropy:** ES training maintains 94% of the base model's output entropy; GRPO training reduces entropy to 71% of baseline after convergence

## Key Contributions

1. A formal analysis showing why GRPO-style RL mechanically reduces output entropy while ES preserves it
2. Empirical measurement of solution diversity using token-entropy and solution-fingerprint metrics across standard benchmarks
3. A practical recipe for ES-based LLM reasoning post-training that achieves GRPO-competitive pass@1 with substantially better pass@k
4. Evidence that the ES–GRPO tradeoff is governed by population size $N$ and perturbation scale $\sigma$, providing tunable knobs for practitioners

## Significance

The paper reframes pass@k as a first-class objective for LLM post-training and provides a principled alternative to RL-based methods for settings where solution diversity matters. For scientific reasoning and exploratory code generation, where finding any correct answer from a diverse pool is more valuable than maximizing a single best-guess score, ES post-training offers a compelling advantage over existing methods.
