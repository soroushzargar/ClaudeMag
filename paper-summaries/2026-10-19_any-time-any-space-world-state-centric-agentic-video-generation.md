# Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation

**arXiv:** 2610.20571  
**Venue:** arXiv preprint, October 2026  
**Authors:** Weijie Wang, Haomin Zhang, Bohan Zhuang, et al. (Zhejiang University, Monash University, BAIR/UC Berkeley, University of Adelaide)  
**Date:** October 14, 2026

---

## One-Line Summary

ATAS maintains two persistent memories—scene geometry and indexed dynamic events—so an agentic video pipeline can correctly revisit any location or replay any event without cross-shot inconsistency.

---

## Problem

Long-form narrative video generation must remain spatially and temporally consistent across nonconsecutive shot requests. A model asked to show the same room from a new angle, or to recall an event that occurred several shots earlier, has no shared spatial frame or time-indexed record available from prompt engineering or recent-frame conditioning alone. This causes visible inconsistencies—duplicated objects, contradictory lighting, location drift—that break narrative coherence.

---

## Key Observation: Cross-Shot State is the Missing Link

The core insight is that the generation process needs a world-state representation that is separate from the film output: a structured record of static geometry (where things are) and dynamic history (what happened and when), persisting across all shot requests regardless of generation order. Retrieval before generation and registration after generation keep this record in sync with the produced content.

---

## Why Existing Approaches Fall Short

- **Prompt-based conditioning:** Passing shot descriptions between calls provides no spatial grounding; the model must re-invent geometry from text at each step.
- **Recent-frame conditioning:** Provides visual continuity for adjacent shots but cannot support true out-of-order revisits or temporally distant event recall.
- **3D-guided methods:** Impose rigid scene constraints that are difficult to maintain across a dynamic narrative with character motion and event changes.
- **Agent planning without persistent state:** Agents that generate shot plans but store no explicit world model lose scene consistency as the plan grows.

---

## Core Method: ATAS

ATAS (Any Time, Any Space) structures the agentic pipeline around two memories:

**Scene Memory:** Stores registered static geometry for each named location. When a new location is first visited, geometry is extracted and registered. All subsequent shots at that location retrieve and condition on this record, ensuring consistent spatial layout regardless of camera angle or elapsed narrative time.

**Event Memory:** Stores dynamic observations indexed by three keys—scene, event, and time. When an action or state change is generated, the resulting clip is reconstructed and registered into the event memory at the appropriate indices. A later shot requesting a replay of that event retrieves the corresponding entry rather than re-generating from scratch.

**Generation Loop:**
1. A screenplay-to-shot parser produces a tagged shot request (location, camera, event time).
2. ATAS queries both memories for supporting evidence.
3. The video generator conditions on the retrieved evidence to produce the shot.
4. A review gate checks the output; if accepted, the clip is reconstructed and registered into memory.
5. Film delivery is independent of memory admission, so a rejected clip can be regenerated without corrupting the state record.

---

## Technical Formulation

Let $\mathcal{S} = \{s_1, \ldots, s_N\}$ be a set of named scenes and $\mathcal{E} = \{e_1, \ldots, e_M\}$ a set of events in the screenplay. Scene memory maps each $s_i$ to a geometric descriptor $G(s_i)$ extracted from the first registered clip at that location:

$$G(s_i) = \text{Reconstruct}(v_{\text{first}}(s_i))$$

Event memory maps $(s_i, e_j, t)$ to a clip descriptor $D(s_i, e_j, t)$ capturing dynamic content at scene $s_i$, event $e_j$, time $t$.

At generation time, for a shot request $r = (s_i, e_j, t, \text{cam})$, the conditioning evidence is:

$$C(r) = G(s_i) \oplus D(s_i, e_j, t)$$

where $\oplus$ denotes the retrieval-and-concatenation operation. The generator produces:

$$v_r = \text{VideoGen}(r, C(r))$$

After review (quality and state-consistency checks), $v_r$ is registered:

$$D(s_i, e_j, t) \leftarrow \text{Reconstruct}(v_r)$$

---

## Learning or Inference Procedure

1. Parse a screenplay into ordered shot requests with scene, event, and time tags.
2. For each request, retrieve from scene and event memory.
3. Condition the video generator on the retrieved evidence and generate the clip.
4. Apply the review gate; if the clip passes, update both memories.
5. Deliver the accepted clip to the film output stream.
6. Repeat until the screenplay is complete.

No additional model training is required; ATAS is a retrieval-and-conditioning framework that wraps any video generation backbone.

---

## What the Paper Claims

- Across four video generators, ATAS improves every reported cross-shot consistency metric.
- ATAS achieves the highest aggregate benchmark score for each backbone.
- On target-image spatial revisit tasks, ATAS outperforms both compared agent-based and 3D-guided methods.
- Human raters gave ATAS the highest scores on all six evaluation criteria across all tested generators.

---

## Experimental Findings

- **Backbones tested:** Four different video generation models covering diffusion-based and autoregressive architectures.
- **Consistency metrics:** Cross-shot appearance, spatial layout, and event recapture measures, all improved by ATAS.
- **Spatial revisit benchmark:** ATAS beats agent baselines and 3D-guided approaches on the target-image spatial revisit task.
- **Human evaluation:** Six criteria including narrative coherence, spatial accuracy, temporal consistency, visual quality, event fidelity, and audio-visual alignment; ATAS tops all six.
- **Out-of-order generation:** ATAS maintains consistency even when shots are generated in non-screenplay order.

---

## Ablations and Interpretation

- **Without scene memory:** Spatial layout drifts across revisit shots; objects disappear or move.
- **Without event memory:** Event recapture degrades to reconstruction from text; temporal positions shift.
- **Without review gate:** Hallucinated geometry gets registered, propagating errors to future shots.
- **Memory size:** Performance saturates once scene memory covers the full set of named locations; event memory benefits scale with granularity of the time index.

---

## Reference

Weijie Wang, Haomin Zhang, Bohan Zhuang, et al. **Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation**. arXiv:2610.20571. October 2026.  
https://arxiv.org/abs/2610.20571
