# ComplexityWorld: Benchmarking Vision-Language Models on Verifiable Visual Decision Making

**arXiv:** 2608.07584  
**Authors:** Not specified (see paper)  
**Submitted:** August 5, 2026  
**Area:** Vision-Language Models, Evaluation, Benchmark

---

## Summary

ComplexityWorld is a benchmark of 390 tasks across 39 domain-inspired visual worlds that tests whether VLMs can extract visual evidence and use it to make decisions that satisfy multi-part constraints — a class of problems distinct from perception or captioning, and one where even the best VLMs currently fail. Only GPT-5.6-Sol reaches 75.6% verifier acceptance rate; all other evaluated models remain below 40%.

## Problem

Real-world AI applications increasingly require systems that can both *perceive* and *decide*: given an image, make a choice whose components collectively satisfy global constraints. For example:

- **Resource allocation:** Given a visual inventory (image of shelves, tools, or items), assign resources to tasks such that all dependencies are met
- **Schedule planning:** Given a visual timetable or calendar, construct a meeting schedule with no conflicts  
- **Spatial arrangement:** Given a map or floor plan, route objects to destinations under capacity constraints

Existing VLM benchmarks test perception (does the model correctly describe what it sees?) or question answering (does the model correctly answer factual questions about an image?), but not *constraint-satisfying decision making from visual evidence*.

The gap matters because even a model that perceives all objects in an image correctly may fail to construct a globally feasible solution — this requires reasoning that integrates visual evidence, domain knowledge, and constraint satisfaction.

## Method

**Task generation pipeline:** Each task is defined by a hidden structured specification: a set of objects, constraints, and a decision space. The specification is rendered into a visual scene (a synthetic image), and an executable verifier checks whether the model's response satisfies all constraints.

**Scale and diversity:** 
- **390 tasks** across **39 domain-inspired visual worlds** (e.g., hospital scheduling, urban routing, factory floor layout, circuit board assembly)
- **29 decision categories** spanning assignment, routing, ordering, partitioning, and resource allocation
- Verifier accepts any *feasible* solution (not just the canonical one), so models are tested on whether they can satisfy constraints, not on whether they match a specific answer

**Modalities tested:** Direct visual inference (model sees only the image), structured-explicit inference (same information given in structured text form), and combined.

## Key Findings

**Overall performance under direct visual inference:**
- All VLMs except GPT-5.6-Sol achieve verifier acceptance rate (VAR) below **40%**
- GPT-5.6-Sol reaches **75.6% VAR**
- Proprietary models substantially outperform open-weight models (gap of 25–40 pp)

**Structured information gap:**
- Performance improves substantially when the same constraint information is made explicit in structured text form rather than requiring visual extraction
- Average improvement from visual to structured mode: **+23.1 pp** VAR
- This "visual constraint extraction gap" accounts for a large fraction of the total failure mode

**Presentation sensitivity:**
- VLMs show sharp performance variation across equivalent visual presentations of the same underlying task
- The same task rendered with different visual style (color scheme, layout, font) changes VAR by up to **18 pp** for the same model
- Suggests that current VLMs have brittle visual representations for constraint-relevant features

## Technical Formulation

The verifier acceptance rate (VAR) is defined as:

$$\text{VAR} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\text{Verify}(r_i, S_i) = \text{FEASIBLE}]$$

where $r_i$ is the model's response to task $i$, $S_i$ is the hidden structured specification, and $\text{Verify}(r_i, S_i)$ runs the domain-specific constraint checker on the parsed response.

Each verifier is formally specified (in a domain-specific language) and executes deterministically, making VAR a reproducible and hallucination-free metric — unlike LLM-as-judge evaluations.

## Significance

ComplexityWorld exposes a fundamental capability gap in current VLMs: the ability to extract multi-part constraints from visual evidence and produce globally feasible solutions. This capability is distinct from perception and from verbal reasoning, and requires tighter integration of visual grounding with constraint satisfaction — an area where current systems significantly lag behind what many real-world deployments would require.
