# Look Before You Reason: Implicit Visual Thinking for Efficient Multimodal Reasoning

**Authors:** (NeurIPS 2026 Poster; research team from leading CV/NLP institutions)  
**arXiv:** 2610.27341 (cs.CV, cs.AI)  
**Submitted:** October 2026  
**Venue:** NeurIPS 2026  

---

## Headline

Before generating a word, a vision-language model learns to silently identify the most decision-relevant image region and amplify its key-value cache entries—achieving explicit-tool-free visual focus that consistently improves multimodal reasoning accuracy.

---

## Key Findings

- **LoRe** (Look-before-Reason) teaches VLMs to predict a relevant image region *before* generating the reasoning chain, without any external tool calls, additional image inputs, or multi-turn interaction.
- Predicted region tokens are used to selectively amplify corresponding key-value cache entries so the decoder naturally attends to those patches during reasoning.
- LoRe is trained with a reinforcement learning objective that rewards both region relevance and final answer correctness.
- Improvements are consistent across three multimodal reasoning benchmarks, with minimal inference overhead (less than 5% extra FLOPs per token on average).
- A new benchmark, **COCO-ObjQA**, is introduced to evaluate fine-grained object-level visual reasoning.

---

## Motivation

Contemporary vision-language models process images by encoding patch embeddings and appending them to the text sequence. When reasoning requires fine-grained visual evidence—identifying a small object, reading embedded text, or comparing spatial relationships—the model must route attention through hundreds of patch tokens without guidance about which patches are decision-relevant.

Two families of solutions exist:
1. **Explicit visual tools:** Zoom, crop, detect, or OCR with dedicated modules or external APIs. These add latency, require tool calling infrastructure, and complicate training.
2. **Multi-turn visual inspection:** The model iteratively re-queries the image. This extends context length and requires specialised multi-turn training.

LoRe proposes a third path: teach the model to perform visual focusing *implicitly*, entirely within a single forward pass, by modifying the key-value cache before reasoning begins.

---

## Method

### Stage 1: Region Prediction Head

A lightweight region-prediction head (a small MLP on top of the visual encoder) is trained to output a bounding box $(x_1, y_1, x_2, y_2)$ in image coordinates. This head is activated at the beginning of the generation (after the question is processed) and before any reasoning tokens are generated.

The region prediction loss is:

$$\mathcal{L}_{\text{region}} = \|\hat{b} - b^*\|_2^2 + \mathcal{L}_{\text{IoU}}(\hat{b}, b^*)$$

where $b^*$ is the ground-truth bounding box for a question-relevant region (derived from visual grounding annotations during training).

### Stage 2: KV Cache Amplification

The image patches whose spatial coordinates fall within the predicted bounding box $\hat{b}$ are identified. For each such patch $p$, the key-value pairs in the cross-attention cache are amplified:

$$K_p \leftarrow \alpha \cdot K_p, \quad V_p \leftarrow \alpha \cdot V_p$$

where $\alpha > 1$ is a temperature parameter (typically 1.5–2.0). This soft amplification focuses decoder attention on the predicted region without hard masking other patches.

### Stage 3: RL Fine-tuning

After supervised pre-training of the region head, the full model (region head + decoder) is fine-tuned with GRPO (Group Relative Policy Optimisation):

$$\mathcal{R}(q, \hat{b}, \hat{a}) = \mathcal{R}_{\text{answer}}(\hat{a}, a^*) + \lambda_{\text{iou}} \cdot \text{IoU}(\hat{b}, b^*)$$

where $\mathcal{R}_{\text{answer}}$ is 1 if the final answer matches the ground truth and 0 otherwise. This joint reward aligns region prediction with answer accuracy end-to-end.

---

## Benchmarks and Results

| Benchmark | LoRe | Base VLM | Delta |
|---|---|---|---|
| MMBench | 83.2 | 80.7 | +2.5 |
| ScienceQA | 91.8 | 89.4 | +2.4 |
| COCO-ObjQA (new) | 76.4 | 69.2 | +7.2 |

The largest gains appear on COCO-ObjQA, which requires precise identification of one object among many in a dense scene—exactly the setting where focused visual attention matters most.

**Ablations:**
- Region prediction without KV amplification (soft selection): +1.1 on MMBench (partial benefit from richer region embeddings).
- KV amplification with random region: -0.4 on MMBench (noise injection hurts).
- Full LoRe (region + amplification + RL): +2.5 on MMBench.

This confirms that both region selection and KV amplification are necessary, and that RL training significantly improves region quality.

**Inference overhead:** The region head adds ~0.3M parameters and ~4% FLOPs; KV amplification is a free element-wise multiply.

---

## COCO-ObjQA Benchmark

COCO-ObjQA is a new benchmark derived from COCO with question-answer pairs that require identifying specific objects in complex multi-object scenes. It includes 12,000 QA pairs across 50 object categories, with ground-truth bounding boxes for the question-relevant object. The benchmark is released alongside the paper to support evaluation of fine-grained visual reasoning in future work.

---

## Significance

LoRe demonstrates that VLMs can learn where to look as a natural part of learning how to reason—without external scaffolding. The KV cache amplification mechanism is architecturally clean and compatible with standard transformer inference. As VLMs are increasingly deployed in applications requiring precise visual reasoning (medical imaging, industrial inspection, robotics), the ability to focus visual attention efficiently and without tool overhead becomes practically important. The RL training paradigm, which rewards region quality through downstream answer accuracy, is a compelling example of end-to-end alignment of perception and reasoning.
