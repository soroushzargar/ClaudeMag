# From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making

**arXiv:** 2609.05149  
**Submitted:** September 4, 2026  
**Authors:** Davide Testa, Hugh Mee Wong, Alessandro Lenci, Bernardo Magnini, Albert Gatt  
**Affiliation:** Fondazione Bruno Kessler (FBK); University of Malta; University of Pisa; and collaborating institutions

---

## Headline Finding

Using layer-wise causal intervention on video-text attention pathways, this paper shows that vision-language models primarily integrate visual information while processing candidate answer tokens (not the question), that nouns act as semantic anchors for cross-modal grounding, and that temporal reasoning exhibits a distinct fragility suggesting VLMs reconstruct sequential visual information poorly — likely conflating linguistic temporal patterns with genuine video understanding.

---

## Key Findings (Pyramid Layer 2)

1. **Visual integration happens at answer tokens.** Causal interventions reveal that visual information is most impactful when injected at the positions where the model processes candidate answer options, not at the question or reasoning tokens.
2. **Nouns are the dominant multimodal grounding sites.** Intervening on noun token positions produces the largest shifts in model decisions, establishing nouns as semantic anchors that bind visual representations to language decisions.
3. **Verbs carry temporal relational load.** While nouns anchor objects, verbs drive processing of temporal relations — motion, causation, and sequence — in temporal reasoning tasks.
4. **Temporal reasoning is uniquely fragile.** Models show a distinct pattern on temporal questions, likely reflecting bias toward linguistic temporal markers rather than genuine frame-by-frame sequential reasoning.

---

## Methodology (Pyramid Layer 3)

### Background: Cross-Modal Information Flow

How visual information propagates from the vision encoder into language-based decisions in VLMs is largely opaque. Most interpretability work examines static image-text tasks. This paper focuses on video, where temporal structure adds an additional layer of complexity: the model must integrate information across frames and map it to temporal linguistic concepts.

### Experimental Framework

The authors construct a video-based generative multiple-choice task where the model selects among 4 candidate answers for questions targeting three reasoning categories:
- **Spatial reasoning:** object location, spatial relations.
- **Causal reasoning:** event causation, action-outcome relations.
- **Temporal reasoning:** event sequence, before/after, duration.

Videos are drawn from standardized evaluation datasets; answer candidates are controlled to be equally fluent and plausible as text.

### Layer-Wise Causal Intervention

The central method is **causal patching** of cross-modal attention:

1. **Run A** (clean run): Process the video-question pair normally and record all attention activations $\mathbf{A}^{(l)}_{t}$ at each layer $l$ and token position $t$.
2. **Corrupted run**: Process a corrupted input (shuffled video frames or masked video tokens) to obtain corrupted activations.
3. **Patching:** Restore the clean attention activations at a specific layer $l^*$ and token position $t^*$ while keeping all other activations from the corrupted run:
$$\mathbf{A}^{(l)}_t = \begin{cases} \mathbf{A}^{(l)}_t[\text{clean}] & \text{if } l = l^*, t = t^* \\ \mathbf{A}^{(l)}_t[\text{corrupted}] & \text{otherwise} \end{cases}$$

The **causal effect** of visual information at layer $l^*$, token $t^*$ is measured by the change in model decision probability:
$$\text{CE}(l^*, t^*) = P(\text{correct} | \text{patched at } l^*, t^*) - P(\text{correct} | \text{fully corrupted})$$

A high causal effect at $(l^*, t^*)$ means that visual information at that layer and position is causally necessary for the correct decision.

### Token-Type Analysis

To identify which token types (question words, answer tokens, nouns, verbs, function words) are the primary integration sites, the authors aggregate $\text{CE}(l^*, t^*)$ by token part-of-speech and position in the input:
$$\text{CE}_{\text{POS}}(\text{pos}) = \frac{1}{|T_{\text{pos}}|} \sum_{t \in T_{\text{pos}}} \sum_{l} \text{CE}(l, t)$$
where $T_{\text{pos}}$ is the set of token positions with part-of-speech label pos.

---

## Technical Formulation

### Causal Effect Normalization

To compare across models and question types, causal effects are normalized:
$$\widehat{\text{CE}}(l^*, t^*) = \frac{\text{CE}(l^*, t^*)}{\text{CE}_{\text{max}}}$$
where $\text{CE}_{\text{max}} = P(\text{correct} | \text{fully clean}) - P(\text{correct} | \text{fully corrupted})$ is the total recoverable effect.

### Temporal Fragility Index

To quantify the unique fragility of temporal reasoning, the authors define:
$$\text{TFI} = \frac{\text{CE}_{\text{spatial}} - \text{CE}_{\text{temporal}}}{\text{CE}_{\text{spatial}}}$$
A high TFI indicates that visual interventions recover less accuracy on temporal questions than on spatial ones, despite equal video content — evidence of temporal understanding fragility.

---

## Experiments

### Models Evaluated
- LLaVA-Video-7B, LLaVA-Video-72B
- Qwen2.5-VL-7B, Qwen2.5-VL-72B
- InternVL3-8B

### Key Quantitative Results

| Metric | Spatial | Causal | Temporal |
|--------|---------|--------|----------|
| Peak CE at answer tokens (normalized) | 0.71 | 0.68 | 0.53 |
| Peak CE at question tokens (normalized) | 0.18 | 0.22 | 0.19 |
| Noun position CE share | 58% | 51% | 44% |
| Verb position CE share | 12% | 24% | 41% |
| Temporal Fragility Index (TFI) | — | — | 0.34 |

The pattern is consistent across all five models, with some variation in peak layer depth.

### Layer Depth of Visual Integration

Visual integration (peak CE) occurs at layers 60–75% of the model depth across all models and question types. Shallow layers show minimal visual effect; the deepest layers show decreasing effect as the model commits to a decision.

---

## Ablations and Interpretation

- **Frame shuffling vs. frame masking corruption:** Both produce similar causal effect patterns, confirming that the ordering information (not just the presence of frames) drives temporal reasoning fragility.
- **Model scale:** Larger models (72B) show higher absolute accuracy but the same qualitative pattern of noun-anchored integration and temporal fragility, suggesting these are fundamental properties of the architecture rather than capacity-limited behaviors.
- **Candidate answer length:** Longer answer options (more tokens) show larger cumulative CE, consistent with the finding that answer token processing is the primary integration site.
- **Temporal fragility interpretation:** High TFI combined with high verb CE share on temporal questions suggests models use linguistic cues about temporal sequences (verb tense, temporal adverbs) rather than reconstructing sequential visual information from frames.

---

## Reference

Davide Testa, Hugh Mee Wong, Alessandro Lenci, Bernardo Magnini, and Albert Gatt. **From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making.** arXiv:2609.05149, September 2026. https://arxiv.org/abs/2609.05149
