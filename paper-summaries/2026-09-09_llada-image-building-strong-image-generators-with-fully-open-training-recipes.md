# LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes

**arXiv:** 2609.03796  
**Submitted:** September 3, 2026  
**Authors:** Chuyan Chen, Haoxing Chen, Kun Chen, Zhenglin Cheng, Long Cui, Ruishan Fang, Zhangxuan Gu, Zhicheng Huang, Zhenzhong Lan, Yuanting Lei, Haoquan Li, Jianguo Li, Rongchuan Li, Sidu Li, Tao Lin, Deyuan Liu, Jiacheng Liu, Lin Liu, Yuxuan Lou, Zhisheng Lu, Yuxin Ma, Shuheng Shen, Peng Sun, Chaoyang Wang, Hongjun Wang, Xiaomei Wang, Yongxin Wang, Chengzhang Wu, Hongru Wu, Jun Xie  
**Affiliation:** inclusionAI (Ant Group)

---

## Headline Finding

LLaDA-Image is a fully open 6B Diffusion Transformer image generator that achieves state-of-the-art Qwen-Image-Bench scores (53.53 English, 53.38 Chinese) by combining staged training (image-only pre-training then paired text-image mid-training), parameter-free RMSNorm, and the Muon optimizer — with all weights, training code, and recipes publicly released, and a distilled LLaDA-Image-Turbo variant enabling high-quality generation in 2–4 sampling steps.

---

## Key Findings (Pyramid Layer 2)

1. **Staged training is essential.** Beginning with image-only pre-training to establish a strong visual generative prior before introducing paired text-image data produces significantly better text-following and image quality than joint training from scratch.
2. **Parameter-free RMSNorm stabilizes DiT training.** Replacing standard LayerNorm or affine RMSNorm with a parameter-free variant throughout the 6B DiT enables stable training at scale without the instability commonly observed in large diffusion transformer runs.
3. **Muon optimizer outperforms AdamW for this architecture.** The Muon optimizer, which applies orthogonal gradient updates, converges faster and to better minima than AdamW for the LLaDA-Image DiT, reducing training compute by approximately 20%.
4. **Full open release enables community replication.** By releasing weights, training code, and detailed stage-by-stage recipes, LLaDA-Image provides the first fully reproducible path to a state-of-the-art open image generator at the 6B scale.

---

## Methodology (Pyramid Layer 3)

### Background: Diffusion Language Models and Unified Generation

LLaDA-Image builds on the **LLaDA** family of diffusion language models, which perform masked discrete diffusion over token sequences. Unlike continuous diffusion (stable diffusion, DALL-E), LLaDA operates in a discrete token space, making it compatible with the language model backbone and enabling unified understanding and generation.

**LLaDA2.0-Mini** is a compact diffusion language model trained on text; LLaDA-Image adds a 6B image generation module while keeping the LLaDA2.0-Mini backbone frozen for the understanding pathway.

### Architecture

**LLaDA-Image** consists of:
- **Vision-Language Understanding Module:** Frozen LLaDA2.0-Mini backbone; processes text inputs and conditions the generation.
- **Image Generation Module (DiT):** A 6B Diffusion Transformer trained from scratch to model masked image tokens.
- **Image Tokenizer:** A VQ-VAE that discretizes images into tokens; shared between the understanding and generation pathways.

The DiT applies parameter-free RMSNorm $\text{RMSNorm}(x) = x / \|x\|_2$ (no learned scale or bias parameters), which eliminates instabilities from scale drift at large model sizes.

### Training Recipe

**Stage 1: Image-only pre-training**  
The DiT is trained on a large corpus of unlabeled images using the masked diffusion objective. The conditioning from the understanding module is disabled; the model learns to model the distribution of natural images without any text. This stage uses 98M real image samples.

**Stage 2: Mid-training (paired text-image)**  
The DiT is fine-tuned on paired (text, image) data with the understanding module providing text conditioning. The model learns to follow text specifications for image content. This stage uses 122M paired samples.

**Stage 3: Instruction fine-tuning**  
A smaller fine-tuning stage on high-quality, instruction-following data (detailed captions, editing instructions). This stage sharpens text-following precision.

**LLaDA-Image-Turbo:** A distilled variant trained with consistency distillation to generate images in 2–4 sampling steps, reducing generation time by 8–16× at minor quality cost.

### Muon Optimizer

The Muon optimizer updates parameters using orthogonalized gradients:
$$\mathbf{G}_t^{\perp} = \text{NewtonSchulz5}(\mathbf{M}_t)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{G}_t^{\perp}$$
where $\mathbf{M}_t$ is the momentum-corrected gradient and NewtonSchulz5 is a 5-iteration Newton-Schulz iteration that orthogonalizes $\mathbf{M}_t$. Orthogonal updates ensure that gradient steps do not collapse or inflate parameter norms, improving stability and convergence rate.

---

## Technical Formulation

### Masked Diffusion Objective

For an image token sequence $x = (x_1, \ldots, x_T)$, the masked diffusion objective trains the DiT to predict masked tokens:
$$\mathcal{L}_{\text{DiT}} = -\mathbb{E}_{t, \mathbf{m}} \left[ \sum_{i : m_i = 1} \log p_\theta(x_i | x_{\setminus m}, c, t) \right]$$
where $\mathbf{m} \in \{0,1\}^T$ is a random mask, $x_{\setminus m}$ is the unmasked tokens, $c$ is the text conditioning from the LLaDA2.0-Mini backbone, and $t$ is the diffusion timestep.

### Parameter-Free RMSNorm

Standard affine RMSNorm:
$$\text{RMSNorm}_{\text{affine}}(x) = \gamma \cdot \frac{x}{\|x\|_2} + \beta$$
Parameter-free variant:
$$\text{RMSNorm}_{\text{free}}(x) = \frac{x}{\|x\|_2}$$

The parameter-free variant eliminates scale drift by fixing the output norm to 1.0, reducing gradient variance during training by removing the learned scaling channel.

### Consistency Distillation (Turbo)

LLaDA-Image-Turbo is trained with:
$$\mathcal{L}_{\text{turbo}} = \mathbb{E}_{x, t, s} \left\| f_\theta(x_t, t) - f_{\theta^-}(x_s, s) \right\|^2$$
where $t > s$ are two timesteps on the same trajectory, $f_\theta$ is the student, $f_{\theta^-}$ is the EMA teacher, and $x_t, x_s$ are noisy versions of the clean image $x$ at their respective noise levels. This enforces that the model produces consistent predictions along self-consistent generation trajectories, enabling convergence in 2–4 steps.

---

## Experiments

### Benchmarks
- **Qwen-Image-Bench:** Comprehensive image generation quality benchmark with English and Chinese prompts; measures photorealism, text alignment, and fine-grained instruction following.
- **GenEval:** Compositional text-to-image generation evaluation.
- **T2I-CompBench:** Text-to-image compositional benchmark.

### Results

| Model | Qwen-Image-Bench (EN) | Qwen-Image-Bench (ZH) | GenEval | T2I-CompBench |
|-------|----------------------|----------------------|---------|---------------|
| DALL-E 3 | 50.1 | 48.7 | 0.61 | 0.584 |
| Stable Diffusion 3.5-L | 49.6 | — | 0.71 | 0.596 |
| Flux-1.1-pro | 51.4 | — | 0.72 | 0.618 |
| **LLaDA-Image (6B)** | **53.53** | **53.38** | **0.74** | **0.631** |
| LLaDA-Image-Turbo (4 steps) | 51.8 | 51.4 | 0.71 | 0.612 |

LLaDA-Image achieves SOTA on Qwen-Image-Bench in both English and Chinese, with LLaDA-Image-Turbo retaining most of the quality at 4-step inference.

### Training Efficiency

| Configuration | Compute (GPU-hours) | Convergence Steps |
|---------------|---------------------|-------------------|
| AdamW baseline | 94,400 | 500K |
| **Muon (ours)** | **75,500** | **380K** |

Muon reduces total training compute by approximately 20% to convergence.

---

## Ablations and Interpretation

- **Stage 1 (image-only) ablation:** Removing image-only pre-training and starting directly with paired text-image training reduces Qwen-Image-Bench score by 3.8 points, confirming that the visual prior must be established before language conditioning is introduced.
- **Parameter-free vs. affine RMSNorm:** Affine RMSNorm causes training instability (loss spikes) at 4B+ parameter scale; parameter-free RMSNorm trains stably throughout.
- **Muon vs. AdamW:** Muon reaches a given validation loss 25% faster; final model quality is 1.1 Qwen-Image-Bench points higher.
- **Turbo step count:** 4 steps achieves 97% of full-model quality; 2 steps retains 94% while being 8× faster.
- **Chinese prompt following:** The bilingual design (LLaDA2.0-Mini backbone processes Chinese natively) gives a strong SOTA margin on Chinese prompts, where English-dominant baselines degrade.

---

## Reference

Chuyan Chen et al. (inclusionAI). **LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes.** arXiv:2609.03796, September 2026. https://arxiv.org/abs/2609.03796
