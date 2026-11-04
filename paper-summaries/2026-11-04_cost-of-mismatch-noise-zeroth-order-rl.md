# The Cost of Mismatch: Noise Amplification in Zeroth-Order Reinforcement Learning

**Authors:** Lianmin Chen, Junbin Qiu, Chenxing Wei, Yao Shu, Kun He  
**arXiv:** 2611.03260  
**Categories:** cs.LG  
**Venue:** NeurIPS 2026 Poster  
**Submitted:** November 2, 2026  
**URL:** https://arxiv.org/abs/2611.03260

---

## Headline

On-policy zeroth-order RL for LLMs suffers from a 1/mu² noise amplification that is fundamental to the mismatched-data setting; matched-data methods like BOLT cancel this term, and the proposed ZO-BOLT achieves faster convergence with lower variance.

## Key Findings

- ZO gradient estimation in the mismatched-data setting incurs a 1/mu² noise penalty that cannot be reduced by decreasing the perturbation scale mu, creating an irreducible noise floor.
- The root cause is not the magnitude of individual evaluation noise but the interaction between two noisy evaluations drawn from different randomness sources.
- BOLT (Boltzmann Targeted SFT) in the matched-data setting avoids this penalty by sharing randomness across the two evaluations, canceling the dominant noise term.
- ZO-BOLT instantiates this matched-data principle for LLM fine-tuning with verifiable rewards, achieving lower gradient variance and faster convergence than standard ZO RL.

## Problem

Zeroth-order optimization methods are attractive for fine-tuning LLMs with RL because they avoid computing gradients through the model, reducing memory from O(parameters) to O(1) for the optimizer state. However, practitioners have observed that ZO RL for LLMs is unstable and slow to converge, in contrast to ZO supervised fine-tuning (SFT) methods like MeZO that work reliably. This paper asks: is there a fundamental difference between ZO SFT and ZO RL that explains this instability?

## Core Analysis

The paper frames ZO RL as estimating a policy gradient with the two-point estimator:
$$\hat{g}_\text{ZO-RL} = \frac{r(\theta + \mu z) - r(\theta - \mu z)}{2\mu} \cdot z$$

where r is a reward function, mu is the perturbation scale, and z is a random direction. In the mismatched-data setting, the two reward evaluations r(theta + mu z) and r(theta - mu z) are computed on independently sampled rollouts, which introduces noise from two independent sources.

The key result is that the variance of this estimator decomposes as:
$$\text{Var}(\hat{g}_\text{ZO-RL}) = \frac{\sigma^2_\text{env}}{2\mu^2} + \text{(lower order terms)}$$

where sigma²_env is the variance of the environment (rollout stochasticity). As mu → 0, the dominant term grows as 1/mu², making it impossible to simultaneously achieve small perturbation bias and small gradient variance.

## The BOLT Solution

Boltzmann Targeted SFT (BOLT) avoids this by operating in the matched-data setting: both reward evaluations share the same rollout seed, so the stochastic noise cancels in the difference. The matched-data estimator variance scales only as sigma²_param/mu², where sigma²_param is the perturbation noise — a much smaller quantity. This cancellation is the key theoretical contribution.

## ZO-BOLT Algorithm

ZO-BOLT adapts BOLT for practical LLM fine-tuning:
1. Sample a random direction z from the parameter space.
2. Generate two rollouts with the same seed under parameters theta + mu z and theta - mu z.
3. Compute the reward difference as the BOLT estimate.
4. Update parameters via SGD using the ZO gradient.

The matched randomness requirement is straightforward to satisfy in practice by fixing the rollout random seed for each pair.

## Experimental Results

- On GSM8K and MATH mathematical reasoning tasks, ZO-BOLT reduces gradient variance by 3-5x compared to standard ZO RL baselines.
- Convergence to 70% accuracy on GSM8K requires 40% fewer forward passes with ZO-BOLT.
- On HumanEval code generation, ZO-BOLT similarly outperforms standard ZO RL approaches.

## Significance

The paper provides a principled theoretical explanation for a widely observed empirical phenomenon (ZO RL instability for LLMs) and derives a simple, low-overhead fix. The 1/mu² amplification is a fundamental property of mismatched-data estimation, not a property of any particular algorithm, suggesting that any ZO RL method should adopt matched-data perturbation as a default design principle.
