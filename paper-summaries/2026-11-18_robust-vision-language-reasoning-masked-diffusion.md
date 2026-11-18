# Towards Robust Vision-Language Reasoning: A Multi-Stage Optimization Pipeline with Masked Diffusion Model

**Authors:** Shunming Liu, Pei Huang, Yajing Sun, Huaxiao Liu
**arXiv:** 2611.15238 (cs.AI, cs.CV)
**Submitted:** 11 November 2026
**Venue:** Preprint

---

## Headline

A multi-stage optimization pipeline applied to vision-language models for autonomous driving, combined with a masked diffusion planning head, yields MD-Drive: a system that improves planning quality by 29.7% and achieves state-of-the-art L2 trajectory error on DriveCoT.

---

## Key Findings

- **MD-Drive** builds a masked diffusion model planning head on top of a vision-language backbone, trained via a multi-stage optimization pipeline.
- Achieves a **PDMS of 88.9** on NAVSIM, outperforming DrivingGPT by 6.5 points and FSDrive by 3.8 points.
- Achieves the best average **L2 trajectory error of 0.78** on DriveCoT, improving over the strongest autoregressive baseline by 1.68 and over MVLAD-AD (best prior discrete-diffusion method) by 0.50.
- The 29.7% improvement over the unoptimized baseline is measured on the primary planning metric.
- Outperforms autoregressive VLM baselines without requiring a separate diffusion planner or mixture-of-experts architecture.

---

## Motivation and Problem

Autonomous driving planning has benefited from large vision-language models that can reason about complex road scenes in natural language. However, standard autoregressive decoding is poorly suited for trajectory generation: it produces trajectories token-by-token, which creates undesirable sequential dependencies and cannot naturally represent the multi-modal distribution of valid future trajectories (e.g., "turn left" vs. "go straight" are equally valid but mutually exclusive).

Discrete diffusion models offer a principled alternative—they generate trajectories by iterative denoising, which naturally captures multi-modality—but prior diffusion planners for autonomous driving have been separate components, not integrated with the vision-language reasoning backbone.

The key question MD-Drive addresses: can a masked diffusion model planning head be integrated into a VLM through a principled multi-stage training pipeline that preserves visual reasoning while learning to generate high-quality trajectories?

---

## Method: Multi-Stage Pipeline and MD-Drive

### Multi-Stage Optimization

The pipeline has three stages:

**Stage 1 (Visual Pre-training):** The VLM backbone is trained on large-scale visual question answering data to build strong scene understanding representations.

**Stage 2 (Driving-Specific Fine-tuning):** The backbone is fine-tuned on paired driving scene descriptions and reference trajectories, learning to associate visual scene features with high-level driving decisions.

**Stage 3 (Diffusion Head Training):** The masked diffusion head is trained on top of frozen backbone representations, learning to denoise masked trajectory tokens using the backbone's scene embedding as conditioning.

### Masked Diffusion Planning Head

The trajectory $\mathbf{x} = (x_1, \ldots, x_T)$ is represented as a sequence of discrete tokens. At each denoising step $k$, a subset of tokens is masked, and the diffusion model predicts the masked positions:

$$q(\mathbf{x}^{(k)} \mid \mathbf{x}^{(0)}) = \prod_t \text{Mask}(x_t, k)$$

$$p_\theta(\mathbf{x}^{(0)} \mid \mathbf{x}^{(k)}, \mathbf{z}_{\text{scene}}) = \prod_t p_\theta(x_t \mid \mathbf{x}^{(k)}, \mathbf{z}_{\text{scene}})$$

where $\mathbf{z}_{\text{scene}}$ is the scene embedding from the VLM backbone. At inference time, the trajectory is initialized as fully masked and iteratively denoised in $K$ steps, with the VLM scene embedding kept fixed.

---

## Experimental Setup and Results

Benchmarks: NAVSIM (planning), DriveCoT (trajectory quality), nuScenes (scene understanding).

**NAVSIM PDMS (planning quality):**
| Method | PDMS |
|---|---|
| DrivingGPT | 82.4 |
| FSDrive | 85.1 |
| MVLAD-AD | 86.3 |
| MD-Drive (ours) | **88.9** |

**DriveCoT L2 error (lower is better):**
| Method | Avg L2 |
|---|---|
| Best autoregressive baseline | 2.46 |
| MVLAD-AD | 1.28 |
| MD-Drive (ours) | **0.78** |

The 29.7% improvement is measured as PDMS gain over the unoptimized VLM backbone (no pipeline stages, no diffusion head).

---

## Significance

MD-Drive demonstrates that the multi-stage pipeline is a key enabler: without it, the diffusion head would be trained on a backbone not yet specialized for driving, and the gains disappear. The integration of masked diffusion into a VLM for planning—rather than as a separate post-processing module—is an architectural contribution that may generalize beyond autonomous driving to other trajectory-generating tasks.
