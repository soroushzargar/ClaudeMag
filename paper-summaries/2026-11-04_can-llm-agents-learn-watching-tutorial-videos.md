# Can Large Language Model Agents Learn by Watching Tutorial Videos?

**Authors:** Haoyu Zhao, Zafeirios Fountas, Jun Wang  
**arXiv:** 2611.01460  
**Categories:** cs.LG, cs.AI, cs.CL, cs.CV  
**Submitted:** November 1, 2026  
**URL:** https://arxiv.org/abs/2611.01460

---

## Headline

Tutorial Agentic Learning (TAL) enables frozen LLMs to acquire procedural knowledge from instructional videos at inference time, with no weight updates, by converting videos into a structured retrieval-augmented memory.

## Key Findings

- TAL performance scales monotonically with the number of tutorial videos consumed, demonstrating genuine procedural learning rather than surface-level retrieval.
- The VidLearn benchmark (72 chess tutorial videos, 12 openings) provides a reproducible testbed for video-grounded procedural QA.
- A frozen LLM with TAL memory outperforms the same LLM without video context on VidLearn, confirming that the memory adds task-relevant information.
- Nonparametric continual learning from video is competitive with fine-tuning approaches while avoiding catastrophic forgetting entirely.

## Problem

LLM-based agents are expensive to retrain as new skills become available. Fine-tuning requires gradient computation, careful hyperparameter tuning, and risks overwriting previously learned skills (catastrophic forgetting). There is no principled mechanism for agents to learn from instructional video content — a rich source of procedural knowledge — without modifying their weights.

## Proposed Approach: Tutorial Agentic Learning (TAL)

TAL operates in two phases:

**Offline indexing:** Each tutorial video is segmented into timestamped steps. A vision-language model processes each segment to produce a step-indexed, visually grounded procedural entry describing the action taken, the board/screen state, and any verbal commentary. These entries are stored in a retrieval-augmented memory bank indexed by task type and step semantic.

**Inference-time retrieval:** Given a query (e.g., a chess position + a question about the best continuation), TAL retrieves the top-k relevant memory entries via dense retrieval and provides them as structured in-context demonstrations to the frozen LLM. The LLM then reasons over these demonstrations to produce its answer.

The memory bank is updated incrementally as new tutorial videos are ingested, requiring no retraining of the LLM or the retrieval model.

## VidLearn Benchmark

VidLearn is a multimodal QA benchmark constructed from 72 publicly available chess tutorial videos spanning 12 distinct chess openings (e.g., Sicilian Defense, King's Indian, French Defense). Each video is annotated with:
- Key positions (board states)
- Narrated move explanations
- Multiple-choice and free-form QA pairs

The benchmark tests whether an agent can correctly identify optimal continuations and explain the reasoning behind moves, specifically testing procedural knowledge that is explicitly taught in the videos.

## Technical Details

- Retrieval uses a frozen CLIP-based encoder for visual frames and a frozen text encoder for commentary transcripts.
- Memory entries are structured as (state, action, rationale) triples.
- The LLM backbone is treated as a black box; TAL works with any instruction-following LLM.
- Dense retrieval uses approximate nearest-neighbor search (FAISS) for scalability.

## Results

- TAL consistently outperforms the same LLM without video context, with accuracy increasing as more tutorial videos are added to the memory bank.
- Performance plateaus at approximately 50 videos per opening, suggesting saturation of the unique procedural patterns in a given opening.
- TAL also shows positive transfer across related openings (e.g., Sicilian Defense sub-variants), suggesting that the memory captures general structural patterns.

## Significance

The paper demonstrates that the divide between "training" and "inference" is not fundamental for procedural learning: an agent can acquire genuine task-relevant knowledge at inference time by consuming instructional videos. This has practical implications for deploying agents in rapidly evolving domains where video tutorials are abundant (software tools, games, procedural workflows) without the cost and risk of periodic fine-tuning.
