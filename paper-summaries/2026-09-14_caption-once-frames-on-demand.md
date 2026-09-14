# Caption-once, Frames-on-Demand: Visual-Need Routing for Budget-Aware Agentic Long Video Understanding

**arXiv:** 2609.11899  
**Submitted:** September 10, 2026  
**Authors:** Weitong Cai, Hang Zhang, Yukai Huang, Yiqiao Xie, Shan Gao, Jiankang Deng, Songcen Xu, Jifei Song, Zhensong Zhang  
**Affiliation:** Imperial College London; Huawei Noah's Ark Lab; and collaborating institutions  
**Venue:** EMNLP 2026 Main Conference

---

## Headline Finding

Caption-once, Frames-on-Demand (CFD) separates long-video understanding into a one-time offline narrative indexing pass and an on-demand online retrieval pass driven by a lightweight Visual-Need Router — achieving state-of-the-art accuracy on long-video QA benchmarks while reducing average API frame calls by 71% compared to dense-frame baselines, enabling edge deployment under tight compute and bandwidth budgets.

---

## Key Findings (Pyramid Layer 2)

1. **Visual-textual duality.** Language memories carry long-range temporal structure better than dense frame sequences — models can answer temporal-structural questions (ordering, causality, scene transitions) from text alone. But pixels are decisive for attribute-level questions (color, appearance, on-screen text) where language captions inevitably abstract away discriminating visual details.
2. **Single offline pass eliminates re-captioning cost.** A CFD deployment captions video once at ingestion time, building a dual-track index reused across all subsequent queries. Prior agentic methods re-caption or re-process frames per query, paying $O(L)$ cost for each question over a video of length $L$.
3. **Visual-Need Router reduces frame retrieval by 71%.** The router classifies each sub-question as "perceptual" (requiring pixel retrieval) or "structural" (answerable from narrative index), triggering actual frame retrieval only for the former. Across EgoSchema, LVBench, and Video-MME, 29% of sub-questions require frame retrieval; the rest are answered from the index.
4. **Dual-track index preserves both temporal scale and granularity.** The event-level story skeleton captures multi-minute narrative arcs; the clip-level micro-log preserves second-level detail including character identities, object states, and quoted speech. The two tracks are jointly indexed and cross-linked for multi-granularity retrieval.

---

## Methodology (Pyramid Layer 3)

### Background: The Edge Long-Video Problem

Deploying video-understanding systems on edge devices (smartphones, in-car systems, wearables) imposes two resource constraints simultaneously:

**Compute budget:** Dense frame encoding over hours of video exceeds available memory and inference latency requirements. Processing one frame per second for a 1-hour video produces 3,600 frames; at typical MLLM token costs, this exceeds the context window of most models.

**Bandwidth budget:** Re-fetching frames from the cloud per query violates latency and cost constraints in edge-cloud deployments. The system must minimize the number of frames transmitted.

Prior approaches fall into two categories:
- **Dense-frame MLLMs** (GPT-4V-Long, LLaVA-1.6-Long): Process all frames in one long context. Accurate but computationally prohibitive at scale.
- **Sparse-frame or memory-based agents** (VideoAgent, EgoVideoQA): Sample frames uniformly or maintain text-only memory. Fast but systematically fail on attribute-level questions.

CFD occupies the middle ground: language index for temporal structure, pixel retrieval only for appearance questions.

### The CFD Framework

**Offline phase (run once at ingestion):**

1. **Clip segmentation:** Video is divided into clips $\{c_1, \ldots, c_K\}$ at scene boundaries using a lightweight shot-detection model (runs locally, no API calls).
2. **Clip captioning:** Each clip $c_k$ is captioned by a compact local MLLM (3B parameter model), producing a micro-log entry $m_k$ containing timestamp range, key actors/objects, described actions, and verbatim on-screen text.
3. **Story synthesis:** A cloud MLLM receives the full micro-log sequence and writes an event-level story skeleton $S$ — a prose summary of 200–500 words covering the main narrative arc, key events, and causal relationships.
4. **Index construction:** The dual-track index $\mathcal{I} = (S, \{m_k\}_{k=1}^K, \{t_k\}_{k=1}^K)$ stores the skeleton, all micro-logs, and their time-stamp maps. This is stored locally at the edge.

**Online phase (per query):**

Given a query $q$, the cloud-side MLLM runs a story-first reasoning loop:

1. **Story pass:** Read $S$ and identify candidate time windows $\mathcal{W}^*$ relevant to $q$.
2. **Micro-log pass:** Read $\{m_k : t_k \in \mathcal{W}^*\}$ and attempt to answer $q$ from text alone.
3. **Visual-Need Router:** Classify the remaining uncertainty as "perceptual" (appearance, visual attribute, on-screen text) or "structural" (temporal order, causality). If structural: return text-based answer. If perceptual: request bounded keyframe retrieval.
4. **Bounded retrieval:** Request at most $R_{\max} = 4$ frames from the relevant time window. Answer the perceptual sub-question and integrate into the final response.

### Visual-Need Router

The router is a binary classifier $\phi: (q, \mathcal{I}) \to \{\text{perceptual}, \text{structural}\}$ trained on a small labeled dataset of 2,400 (question, index) pairs annotated for visual necessity.

Architecture: A 350M parameter BERT-style encoder on the concatenation of the question and the relevant micro-log excerpt.

The router achieves 91.4% accuracy on a held-out test set of 600 pairs. The cost of running the router is negligible (< 1ms on CPU) compared to frame retrieval (200–800ms round-trip cloud latency).

---

## Technical Formulation

Let $V = \{f_1, \ldots, f_N\}$ be a video with $N$ frames, $q$ a user query, and $\mathcal{A}$ the correct answer. The CFD pipeline computes:

$$\hat{\mathcal{A}} = \text{MLLM}\left(q,\, S,\, \{m_k : t_k \in \mathcal{W}^*(q, \mathcal{I})\},\, \{f_j : j \in \mathcal{R}(q)\}\right)$$

where $\mathcal{W}^*$ is the time-window retrieval function, and:
$$\mathcal{R}(q) = \begin{cases} \arg\max_{|J| \leq R_{\max}} \mathrm{Relevance}(q, \{f_j\}_{j \in J}) & \text{if } \phi(q, \mathcal{I}) = \text{perceptual} \\ \emptyset & \text{otherwise} \end{cases}$$

**Cost analysis.** Total frame API calls per query: $|\mathcal{R}(q)| \leq R_{\max} \cdot \mathbf{1}[\phi(q) = \text{perceptual}]$. Expected calls per query $= R_{\max} \cdot p_{\text{perc}}$ where $p_{\text{perc}} = 0.29$ is the empirical perceptual question rate, giving expected calls of $4 \times 0.29 = 1.16$ frames per query versus $\sim 4$ for dense-frame agents.

---

## Experiments

### Benchmarks
- **EgoSchema (500 questions):** Egocentric activity recognition over 3-minute clips.
- **LVBench (1,000 questions):** Long-video QA over 1–4 hour videos across diverse genres.
- **Video-MME (Long split, 900 questions):** Multi-task long video evaluation.

### Main Results

| Method | EgoSchema | LVBench | Video-MME-Long | Avg. Frames/Q |
|--------|-----------|---------|----------------|---------------|
| GPT-4V-Long (dense) | 72.4% | 58.3% | 64.1% | 64.0 |
| VideoAgent | 68.9% | 52.7% | 59.8% | 18.4 |
| EgoVideoQA | 70.3% | 54.1% | 61.2% | 22.0 |
| **CFD (ours)** | **74.8%** | **60.9%** | **66.7%** | **1.16** |

CFD improves over the best prior method by 2.4–2.8 accuracy points while using 15–55× fewer frames per query.

---

## Ablations and Interpretation

- **Without Visual-Need Router (always retrieve):** Accuracy improves by 0.9 points but frame cost increases 3.4×, confirming the router's accuracy-efficiency tradeoff is favorable.
- **Without story skeleton (micro-logs only):** Accuracy drops by 5.3 points on LVBench, showing the story skeleton is essential for multi-scene temporal reasoning.
- **Without micro-logs (story only):** Accuracy drops by 3.1 points on EgoSchema, confirming micro-logs are needed for fine-grained clip-level questions.
- **Single-track index:** Either track alone scores 4–8 points below the dual-track system, confirming the complementarity of temporal scales.

---

## Reference

Weitong Cai, Hang Zhang, Yukai Huang, Yiqiao Xie, Shan Gao, Jiankang Deng, Songcen Xu, Jifei Song, and Zhensong Zhang. **Caption-once, Frames-on-Demand: Visual-Need Routing for Budget-Aware Agentic Long Video Understanding.** arXiv:2609.11899, EMNLP 2026. https://arxiv.org/abs/2609.11899
