# TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM

**Authors:** Hengyi Xie, Chenfei Yao, Xianjin Wu, Xuanyang Xi, Yiping Tang, Di Xu, Yingying Zhu, Dingkang Liang, Xiang Bai, Han Ding  
**Affiliation:** Huazhong University of Science and Technology; Huawei Technologies Co. Ltd  
**Venue:** arXiv:2607.27205  
**Date:** July 30, 2026  
**URL:** https://arxiv.org/abs/2607.27205

---

## Summary

Vision-language-action (VLA) models have emerged as a compelling paradigm for generalist robot policies: a pre-trained vision-language model is extended with an action head, leveraging broad visual and linguistic pretraining for zero-shot and few-shot manipulation. However, the dominant paradigm routes all information through a large language model—an LLM backbone that processes visual tokens in its attention layers at every inference step—creating a severe compute bottleneck that makes real-time deployment on consumer hardware infeasible. TurboVLA proposes a structural fix: decouple visual and language encoding, exchange information through a lightweight bidirectional cross-attention module, and decode actions directly. The result is a 0.2B parameter policy running at 32 Hz on a single RTX 4090 with under 1 GB VRAM.

## Problem

Existing VLA architectures (OpenVLA, Pi-0, etc.) adopt a V→L→A pathway: visual observations are tokenized and projected into the LLM's embedding space, which then processes the combined token sequence through many transformer layers before an action head decodes from the final representation. This design has two practical consequences:

1. **Compute cost scales with LLM size.** A 7B-parameter LLM backbone processes hundreds of visual tokens per action step, costing hundreds of milliseconds per inference on a single GPU.
2. **Memory cost is dominated by the LLM weights.** Even at 4-bit quantization, a 7B model occupies ~3.5 GB VRAM, preventing deployment on consumer robotics platforms.

The asymmetry is stark: the actual information flow for action prediction requires relatively simple alignment between visual observations and a language goal, yet the entire LLM backbone must participate in every forward pass.

## Method: V+L→A Direct Mapping

TurboVLA decouples the three components:

1. **Visual encoder:** A compact ViT-S/16 processes the observation image into a sequence of patch tokens $V \in \mathbb{R}^{N_v \times d_v}$.
2. **Language encoder:** A frozen CLIP text encoder processes the language instruction into a sequence of tokens $L \in \mathbb{R}^{N_l \times d_l}$.
3. **Bidirectional Vision-Language Interaction (BiVLI):** A small transformer with bidirectional cross-attention that lets $V$ attend to $L$ and $L$ attend to $V$ for two alternating rounds. This produces aligned representations $V' \in \mathbb{R}^{N_v \times d}$ and $L' \in \mathbb{R}^{N_l \times d}$.
4. **Action decoder:** A lightweight MLP that takes the global pooled representation $[\text{pool}(V') \oplus \text{pool}(L')]$ and predicts an action chunk (a fixed-horizon sequence of continuous control signals).

No autoregressive LLM backbone is used during inference. The BiVLI module contains only $\sim$50M parameters; the visual encoder and action decoder together add another $\sim$150M, for a total of $\sim$0.2B parameters.

## Technical Formulation

The BiVLI module alternates between visual-to-language and language-to-visual cross-attention. For round $r$:

$$V^{(r)} = V^{(r-1)} + \text{MHA}_{V \leftarrow L}\!\left(Q=V^{(r-1)},\; K=L^{(r-1)},\; V=L^{(r-1)}\right)$$

$$L^{(r)} = L^{(r-1)} + \text{MHA}_{L \leftarrow V}\!\left(Q=L^{(r-1)},\; K=V^{(r)},\; V=V^{(r)}\right)$$

where $\text{MHA}$ denotes multi-head attention. After $R=2$ rounds, the pooled representation

$$z = \text{pool}(V^{(R)}) \oplus \text{pool}(L^{(R)})$$

is passed to the action decoder to predict an action chunk $a_{1:H} \in \mathbb{R}^{H \times d_a}$ for a horizon of $H$ steps.

Training uses a behavior cloning loss on expert demonstrations:

$$\mathcal{L}_{\text{BC}} = \sum_{h=1}^{H} \left\| \hat{a}_h - a_h^* \right\|_2^2$$

where $a_h^*$ is the demonstrated action at horizon step $h$.

## Learning Procedure

TurboVLA is trained in two stages:

1. **Pretraining:** The visual encoder is initialized from CLIP ViT-S/16. The BiVLI module is randomly initialized and trained jointly with the action decoder on a large offline dataset of robot demonstrations (OpenX-Embodiment).
2. **Task-specific fine-tuning:** The full model (including visual encoder) is fine-tuned on task-specific demonstrations for a target environment.

The language encoder is kept frozen throughout training to leverage the rich semantic structure of the pretrained CLIP text encoder.

## What the Guarantee Says

The authors analyze the information-theoretic bottleneck created by LLM-centric V→L→A architectures. The LLM processes the full visual token sequence autoregressively, so compute is $O(N_v^2)$ per forward pass (due to attention). TurboVLA's BiVLI module uses cross-attention, so visual tokens attend to language tokens in $O(N_v \cdot N_l)$ and vice versa—strictly lower complexity. For typical values ($N_v = 196$ patch tokens, $N_l \approx 30$ language tokens), the interaction cost is $\sim30\times$ lower than full self-attention over the combined sequence.

## Experimental Findings

**LIBERO benchmark (5 suites: LIBERO-Spatial, -Object, -Goal, -Long, -PRO):**

| Model | Params | Avg. Success | Latency (ms) | VRAM (GB) |
|---|---|---|---|---|
| OpenVLA-7B | 7B | 96.1% | 218 | 14.0 |
| Pi-0 (distilled) | 3.3B | 97.2% | 82 | 6.4 |
| TurboVLA (ours) | **0.2B** | **97.7%** | **31.2** | **0.9** |

TurboVLA matches or exceeds the success rate of models 15–35× larger, while running 7–35× faster and consuming 7–16× less memory.

**Cross-embodiment generalization:** Fine-tuned on three robot morphologies (Franka, xArm, UR5), TurboVLA shows less catastrophic forgetting than OpenVLA due to its frozen language encoder acting as a semantic anchor.

## Ablations and Interpretation

**BiVLI rounds:** Reducing from $R=2$ to $R=1$ drops average success by 3.2%; increasing to $R=4$ adds only 0.4% while doubling BiVLI latency. Two rounds is the sweet spot.

**Encoder size:** Replacing ViT-S/16 with ViT-B/16 improves success by 1.1% but doubles VRAM. The trade-off is appropriate for resource-constrained deployment.

**Action chunk horizon:** $H=10$ gives the best task completion rate; shorter horizons ($H=4$) reduce success by 4.1% due to higher control frequency jitter, and longer horizons ($H=20$) reduce it by 2.3% due to compounding prediction error.
