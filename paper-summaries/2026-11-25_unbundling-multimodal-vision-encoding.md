# Unbundling Multimodal Vision Encoding

**Authors:** Ellis Brown, Chris Hoang, Shang-Wen Li, Mike Lewis, Marjan Ghazvininejad, Luke Zettlemoyer, Lucas Beyer, Mengye Ren, Jakob Verbeek, Saining Xie, Hu Xu
**Affiliation:** NYU; Meta FAIR
**arXiv:** 2611.17334 (cs.CV, cs.LG)
**Submitted:** ~18 November 2026

---

## Headline

Standard vision encoders bundle three distinct design decisions—visual pretraining prior, dedicated vision capacity, and placement within the model—into one opaque component; separating them reveals that dedicated vision capacity trained jointly with the backbone and inserted early is the load-bearing ingredient, while upstream visual pretraining is not.

---

## Key Findings

- A vision encoder typically conflates three choices: (1) a **visual prior** from upstream pretraining, (2) **dedicated vision capacity** (additional parameters devoted to processing images), and (3) **placement** of that capacity (pre-fusion or inside the backbone).
- Placing dedicated vision capacity **early inside the language backbone**, trained jointly from scratch, matches the performance of a separately pretrained pre-fusion encoder.
- Jointly trained early capacity **reduces time to first token** compared to a pre-fusion encoder, because it eliminates a separate forward pass before fusion.
- An encoder-free ablation (a 1.7B model with no dedicated vision parameters) trails the best configuration by **5.4 points** on average VQA, confirming dedicated capacity is essential.
- The results hold across general vision-language benchmarks, visual hallucination tasks, and vision-centric tasks.

---

## Motivation and Problem

Multimodal language models almost universally attach a separately pretrained visual encoder—such as a CLIP ViT—to a language backbone. This design choice is rarely questioned: it is assumed that the upstream visual pretraining provides a beneficial inductive bias that bootstraps multimodal learning.

The authors observe that this assumption bundles three separate hypotheses:
1. The model needs a visual prior learned from image-only or image-text data.
2. The model needs dedicated parameters for processing visual information.
3. Vision processing must happen in a separate module before language-vision fusion.

None of these hypotheses has been cleanly isolated. Prior ablation work has changed one while holding others constant, but no paper systematically varied all three within a single controlled experimental framework.

If hypothesis 1 is wrong (the prior is not essential), then the standard encoder introduces an unnecessary dependency on a large pretrained component. If hypothesis 3 is wrong (placement matters but prior pretraining does not), then jointly trained vision modules inside the backbone may work equally well. The practical stakes are high: eliminating upstream visual pretraining simplifies the training pipeline, reduces data requirements, and potentially reduces latency at inference.

---

## Method

### Three-Way Factorization

The paper defines three orthogonal design axes:

1. **Prior**: Whether dedicated vision capacity is initialized from upstream visual pretraining (e.g., CLIP) or from random initialization (or from the language model's own weights).
2. **Capacity**: Whether the model has dedicated parameters for vision processing (yes/no), and how many.
3. **Placement**: Whether dedicated capacity sits before fusion (pre-fusion encoder), at the fusion point, or inside the backbone layers.

Each combination defines a distinct architecture variant. The authors train each variant with matched compute and parameter budget and evaluate on the same benchmark suite.

### Architectures Evaluated

The paper tests five main configurations:
- **Baseline**: CLIP-pretrained ViT pre-fusion encoder (standard VLM architecture)
- **Early-joint**: Dedicated capacity inserted in early backbone layers, trained from scratch jointly with the LLM
- **Late-joint**: Dedicated capacity inserted in late backbone layers, trained jointly
- **No-prior**: CLIP-initialized encoder but fine-tuned jointly (separates prior from capacity)
- **Encoder-free**: No dedicated vision parameters; raw image tokens processed directly by the backbone

---

## Technical Details

The vision capacity in the **early-joint** configuration consists of alternating vision-specific and shared attention blocks inserted into the first third of the backbone transformer layers. The vision-specific blocks attend only to image tokens; the shared blocks attend to both image and text tokens. This keeps the visual processing close to the token-embedding stage, allowing the model to build visual representations before language context accumulates.

The authors argue that the advantage of early placement is information bottleneck: the backbone can establish a compact visual representation early, before language-side information contaminates or overrides visual signals in later layers.

Time-to-first-token is reduced because the early-joint configuration eliminates the forward pass through a separate encoder (ViT). The image tokens enter the backbone directly, and their representation is built up progressively as tokens flow through early layers.

---

## Experimental Findings

Training was conducted on a 1.7B backbone. All configurations use the same training data and same compute budget (FLOPs matched to within 2%).

| Configuration | Avg VQA | Visual Hallucination | Vision-Centric | TTFT (rel.) |
|---|---|---|---|---|
| Baseline (CLIP pre-fusion) | 72.1 | 68.3 | 64.5 | 1.00× |
| Early-joint (no prior) | **71.8** | **68.1** | **64.9** | **0.87×** |
| Late-joint (no prior) | 69.4 | 66.2 | 62.7 | 0.90× |
| No-prior encoder | 70.6 | 67.1 | 63.8 | 1.02× |
| Encoder-free | 66.4 | 63.8 | 60.1 | 0.78× |

The early-joint configuration matches the CLIP pre-fusion baseline to within 0.3 points on average VQA while running 13% faster at inference. The encoder-free ablation lags by 5.4 points, confirming dedicated capacity is necessary.

---

## Ablations and Interpretation

- **Prior vs. capacity:** Comparing "no-prior encoder" (capacity with random init) to "baseline" (capacity with CLIP init) shows only a 1.5-point difference, suggesting visual pretraining contributes marginally once dedicated capacity is present.
- **Early vs. late placement:** Early-joint outperforms late-joint by 2.4 points, confirming that vision capacity is more effective when placed before language context is established.
- **Parameter budget:** Increasing the size of the early-joint capacity (from 10% to 20% of backbone parameters) yields modest gains (+0.8 points) with diminishing returns beyond 15%.

---

## Significance

This paper is a principled audit of a design choice that has shaped multimodal model development for years. The result that visual pretraining prior is not the essential ingredient—dedicated capacity and early placement are—implies that multimodal training pipelines can be simplified substantially. Removing the dependency on a large pretrained vision encoder reduces data requirements (no image-only pretraining needed), reduces the number of moving parts in the training stack, and delivers a small latency improvement in deployment. For teams building multimodal systems from scratch, this paper provides concrete architectural guidance backed by systematic ablation.
