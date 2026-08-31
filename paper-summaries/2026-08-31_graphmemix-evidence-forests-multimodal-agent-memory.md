# GraphMemix: Query-Aware Evidence Forests for Long-Term Multimodal Agent Memory

**arXiv:** 2608.26983  
**Authors:** Geng Li, Yuhao Wang, Dong Li, Jianye Hao, Yuxin Peng  
**Submitted:** August 27, 2026  
**Area:** Multimodal AI, Agent Memory, Graph Retrieval, Long-Horizon Reasoning

---

## Summary

GraphMemix is a memory retrieval system for long-horizon multimodal agents that dynamically constructs query-aware evidence forests from a heterogeneous memory graph. Unlike flat dense retrieval or static graph schemas, GraphMemix adapts its retrieval structure to each query's intent, enabling multi-hop evidence aggregation across text, image, and video modalities. On long-horizon multimodal QA and agent navigation benchmarks, GraphMemix substantially improves retrieval accuracy over dense retrieval and static graph baselines while maintaining sub-second query latency.

## Problem

Multimodal agents operating over extended interaction histories — multiple images, video clips, and conversational turns — face three compounding challenges in memory retrieval:

1. **Modal heterogeneity:** Evidence relevant to a query may span text, images, and video frames, requiring unified retrieval across incompatible feature spaces
2. **Multi-hop dependencies:** Answering complex queries often requires chaining multiple memory entries (e.g., object A appeared in image B, which was described in message C)
3. **Query intent variation:** The same memory graph must support different retrieval patterns depending on the query — sometimes breadth-first to gather diverse evidence, sometimes depth-first to trace a causal chain

Existing approaches fail to address all three simultaneously. Dense retrieval (vector similarity) handles heterogeneity but not multi-hop dependencies. Graph memory systems capture dependencies but impose static retrieval schemas that cannot adapt to query intent.

## Method

GraphMemix maintains a **heterogeneous memory graph** $G = (V, E)$ where:
- Nodes $V$ represent memory items (text segments, image embeddings, video clip summaries)
- Edges $E$ represent typed relationships (temporal, causal, co-reference, spatial)

For each incoming query $q$, GraphMemix constructs a **query-aware evidence forest** through three steps:

**Step 1 — Seed node retrieval:** A dual-encoder retrieves top-$k$ seed nodes from $V$ via cross-modal similarity between the query embedding and node embeddings.

**Step 2 — Forest construction policy:** A lightweight GNN-based policy $\pi_\phi$ takes the query embedding, seed node embeddings, and local subgraph structure as input, and selects which edges to expand from each seed node. The policy outputs a tree rooted at each seed, where each edge selection is conditioned on the current query intent embedding. This transforms the multi-hop retrieval problem into a sequence of local edge decisions rather than requiring global graph search.

**Step 3 — Evidence aggregation:** Nodes in the forest are encoded by a multimodal transformer that attends across modalities. The resulting evidence representations are ranked by a learned relevance scorer and the top-$m$ are passed to the agent's context window.

**Training:** The forest construction policy is trained with sparse rewards from downstream task accuracy, using REINFORCE with a baseline derived from dense retrieval performance. No ground-truth multi-hop paths are required.

## Technical Formulation

Let $h_v$ be the embedding of node $v$, $h_q$ the query embedding. The policy at node $v$ computes expansion probabilities over outgoing edges $(v, u)$:

$$P_\phi(u | v, q) = \sigma\left(W_\phi [h_v \| h_u \| h_q \| r_{vu}]\right)$$

where $r_{vu}$ is the edge relation type embedding and $\|$ is concatenation. The policy selects the top-$b$ expansions (branch factor) at each node, constructing a tree of depth $d$ in $O(b^d \cdot |V_{\text{seed}}|)$ node evaluations — tractable because $b$ and $d$ are small (typically $b=3, d=2$).

The forest is constructed in parallel across seed nodes, yielding a disjoint union of evidence trees that is then merged by removing duplicate nodes.

## Results

Evaluated on three benchmarks:

**EgoSchema-Long:** Extended video QA over 45-minute egocentric video streams. GraphMemix improves accuracy by +7.2% over dense retrieval and +4.1% over the best static graph baseline.

**MMAgent-Bench:** Multimodal agent navigation requiring memory of visited locations, objects, and events. Task completion rate improves by +11.3% over dense retrieval.

**MultiModal-NarrativeQA:** Long-form QA over image-text interleaved narratives. GraphMemix improves F1 by +6.8 points.

**Ablations:**
- Removing the query-conditioned forest construction policy (using static BFS expansion) degrades performance by −4.3% on average, confirming that query-aware structure is critical
- Removing cross-modal aggregation (using modality-specific encoders without cross-attention) degrades by −3.1%
- Latency: 0.32s average per query on a single A100 GPU for a 10K-node memory graph

## Key Contributions

1. A query-aware forest construction mechanism that adapts multi-hop retrieval structure to each query's intent, learned without ground-truth path supervision
2. A unified framework for heterogeneous multimodal memory that handles text, image, and video in a single graph structure
3. Substantial accuracy gains over dense retrieval and static graph memory on three long-horizon benchmarks, at practical query latency
4. Evidence that the query-conditioned forest structure (rather than graph connectivity alone) is the key factor driving improvements

## Significance

As multimodal agents are deployed in longer and more complex interaction settings, memory retrieval quality becomes a bottleneck for agent capability. GraphMemix demonstrates that dynamically structuring retrieval to match query intent — rather than relying on a fixed retrieval schema — substantially improves both accuracy and multi-hop evidence aggregation. The approach is practical (low latency, no ground-truth path annotation) and applicable to any agent architecture that maintains a persistent memory store.
