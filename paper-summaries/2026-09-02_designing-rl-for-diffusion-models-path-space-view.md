# Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View

**arXiv:** 2608.14430  
**Authors:** Yixian Xu, Yuanrui Zhang, Shengjie Luo, Liwei Wang, Di He  
**Affiliation:** Peking University; ByteDance Seed  
**Submitted:** August 2026  
**Area:** Diffusion Models, Reinforcement Learning, Generative Models, Theory

---

## Summary

Reinforcement learning fine-tuning of diffusion models aligns their outputs with task-specific rewards, but the existing algorithmic landscape is fragmented: reverse-trajectory methods apply RL via discretized likelihood ratios over the reverse SDE, while forward-matching methods train on reward-labeled noising trajectories, with no shared theoretical basis connecting them. This paper shows that both families emerge from a single path-space importance-sampling principle. The unification yields a concrete design space—organized by value-gradient estimation, weight functions, and sampling strategy—from which the authors derive a new multi-sample KDE value-gradient estimator that outperforms prior methods on image and vision-language model alignment tasks.

## Problem

Diffusion models define a generative process through a forward noising SDE and a learned reverse SDE. Fine-tuning for reward alignment means biasing the reverse process toward samples with high reward $R(x_0)$ without destroying the model's coverage of the data distribution.

Several RL algorithms for diffusion models exist, including:
- **DDPO** and **DPO-Diffusion**: treat each denoising step as a policy action and apply PPO-style probability-ratio clipping over the reverse trajectory.
- **ReFL** and **AlignProp**: differentiate through the reverse SDE to compute a direct reward gradient.
- **Flow-matching variants**: apply reward-weighted updates on forward-noised samples.

These methods use different objective formulations, and it is unclear which design choices matter, how they relate theoretically, or how to derive new methods systematically.

## Method

The paper derives all major diffusion-RL methods from a single path-space principle. Consider the reverse SDE as generating a trajectory $\tau = (x_T, x_{T-1}, \ldots, x_0)$. The goal is to importance-weight the trajectory distribution $p_\theta(\tau)$ toward the target distribution $p^*(\tau) \propto p_\theta(\tau) R(x_0)$.

The path-space importance weight is:

$$w(\tau) = \frac{p^*(\tau)}{p_\theta(\tau)} = \frac{R(x_0)}{\mathbb{E}_\tau[R(x_0)]}$$

Taking the gradient of the policy objective $J(\theta) = \mathbb{E}_{p_\theta}[w(\tau) R(x_0)]$ with respect to $\theta$ yields:

$$\nabla_\theta J(\theta) = \mathbb{E}_{p_\theta}\left[R(x_0) \nabla_\theta \log p_\theta(\tau)\right]$$

This is a standard policy gradient, but applied to the full denoising trajectory rather than a single-step action. The paper shows that:
- **Reverse-trajectory RL methods** approximate this gradient by step-wise importance ratios $\prod_t \rho_t$ where $\rho_t = p_\theta(x_{t-1}|x_t) / p_{\theta_\text{old}}(x_{t-1}|x_t)$.
- **Value-gradient methods** (ReFL, AlignProp) re-express the gradient as a value function gradient $\nabla_\theta V^\pi(x_T)$ and compute it by backpropagating through the reverse SDE.
- **Forward-matching methods** arise as reparametrization of the same path-space gradient under the ELBO decomposition.

## Technical Formulation

The value function of the denoising process starting from $x_t$ is:

$$V^\pi(x_t) = \mathbb{E}_{p_\theta(\tau_{t:0} | x_t)}[R(x_0)]$$

The value-gradient form of the policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x_T}[\nabla_\theta V^\pi(x_T)]$$

The paper proposes a **multi-sample KDE estimator** of $V^\pi(x_t)$:

$$\hat{V}(x_t) = \frac{\sum_{i=1}^K R(x_0^{(i)}) K_h(x_0 - x_0^{(i)})}{\sum_{i=1}^K K_h(x_0 - x_0^{(i)})}$$

where $K$ rollouts from $x_t$ are used and a kernel $K_h$ smooths the value estimate. This reduces variance compared to single-sample REINFORCE estimates and does not require a separate critic network.

Weight functions across the design space are parameterized as:

$$f(w(\tau)) \in \{w, \text{clip}(w, 1-\epsilon, 1+\epsilon), \mathbb{1}[w > c]\}$$

corresponding to vanilla policy gradient, PPO clipping, and binary advantage filtering respectively.

## Experimental Results

Evaluated on:
- **Image generation (SD3.5-M):** Reward alignment on aesthetic quality and compositional faithfulness benchmarks. The KDE estimator outperforms single-sample REINFORCE by 8.3% on human preference alignment.
- **Vision-language model (Qwen-Image):** Reward fine-tuning on VQA accuracy. KDE estimator improves over AlignProp by 4.1% absolute on the target reward.
- **Variance analysis:** The KDE estimator reduces gradient variance by 3.2× compared to single-sample baselines, confirming the theoretical variance-reduction analysis.

## Implications

The unified design space allows systematic comparison and ablation of diffusion-RL algorithms. Practitioners can select weight functions, gradient estimators, and sampling strategies from a principled menu rather than treating each method as a separate invention. The KDE value estimator is immediately applicable to any diffusion model reward fine-tuning pipeline.

## Reference

Yixian Xu, Yuanrui Zhang, Shengjie Luo, Liwei Wang, Di He. **Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View.** arXiv:2608.14430, August 2026.  
https://arxiv.org/abs/2608.14430
