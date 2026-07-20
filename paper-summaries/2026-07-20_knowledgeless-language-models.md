# Knowledgeless Language Models: Suppressing Parametric Recall for Evidence-Grounded Language Modeling

**arXiv:** 2607.12831  
**Date:** July 14, 2026  
**Authors:** Roi Cohen, Yvan Carré, Nick Lechtenbörger, Hendrik Droste, Lucas Kerschke, Russa Biswas, Gerard de Melo, Jan Buys  
**Affiliation:** Hasso Plattner Institute, University of Cape Town  
**URL:** https://arxiv.org/abs/2607.12831

---

## Headline

KLLMs (Knowledgeless Language Models) are pretrained on entity-anonymized corpora to suppress parametric factual recall, producing models that are less confident about facts they cannot see but significantly better at using facts they can — a pretraining-level intervention for hallucination reduction.

---

## Key Findings

- Anonymizing named entities in pretraining corpora substantially reduces closed-book factual recall without degrading linguistic competence.
- KLLMs consistently outperform matched baselines on contextual QA, fact verification, and hallucination detection benchmarks — tasks where the answer is present in the context.
- The effect is consistent across multiple model scales (125M, 350M, 1.3B parameters), suggesting it is a pretraining-level phenomenon rather than an artifact of scale.
- KLLMs show improved calibration: they express higher uncertainty on questions that require world knowledge not in context, rather than confidently hallucinating.
- Entity anonymization has a negligible effect on language modeling perplexity, indicating that linguistic structure is learned independently of entity-specific facts.

---

## Methodology

KLLMs are pretrained on a modified corpus where named entities (persons, organizations, locations, dates, and named products) are replaced with consistent anonymized tokens. The anonymization is consistent within each document (the same entity always maps to the same placeholder token) but random across documents.

1. **Anonymization pipeline:** A named entity recognition (NER) model identifies spans; each unique entity is replaced with a type-annotated token (e.g., `[PERSON_42]`, `[ORG_7]`).
2. **Cross-document consistency:** Within a single document, `[PERSON_42]` refers to the same person; across documents, the mapping resets, preventing the model from associating placeholders with real-world referents.
3. **Linguistic structure preserved:** Syntax, coreference, and discourse structure are unchanged; only entity identity is obscured.
4. **Standard training:** The model is trained with standard autoregressive cross-entropy loss; no special loss terms or training procedures.

---

## Technical Details

- **Entity anonymization coverage:** ~12% of tokens in typical web corpora are entity-linked; these are the tokens replaced.
- **Baseline comparison:** Each KLLM is compared to a model trained on the same corpus without anonymization, using identical hyperparameters and compute budgets.
- **Evaluation benchmarks:** NaturalQuestions (closed-book and open-book), TriviaQA (closed-book and open-book), FEVER (fact verification), HaluEval (hallucination detection), TruthfulQA.
- **Calibration metric:** Expected Calibration Error (ECE) on open-book vs. closed-book settings.

---

## Experimental Findings

- Closed-book NaturalQuestions: KLLM-1.3B scores 6.2% EM vs. 18.7% for baseline (intended reduction; entity-specific facts are suppressed).
- Open-book NaturalQuestions: KLLM-1.3B scores 43.1% EM vs. 38.4% for baseline (+12.2% relative improvement).
- FEVER (fact verification with evidence): KLLM-1.3B: 83.7% accuracy vs. 79.2% for baseline.
- HaluEval (hallucination detection): KLLM-1.3B: 72.4% vs. 65.1% for baseline.
- TruthfulQA: KLLM-1.3B: 48.3% vs. 41.7% for baseline.
- Perplexity on held-out text: KLLM-1.3B: 14.3 vs. 14.1 for baseline (negligible difference).

---

## Ablations and Interpretation

- **Partial anonymization:** Anonymizing only persons (vs. all entities) yields roughly half the improvement on HaluEval, suggesting that organization and location knowledge contributes substantially to parametric recall.
- **Cross-document randomization vs. fixed mapping:** Using a fixed mapping (the same entity always gets the same placeholder across all documents) partially re-introduces factual associations; the random-per-document approach is essential.
- **Effect on downstream fine-tuning:** KLLMs fine-tuned on Wikipedia-based QA datasets close much of the closed-book gap while retaining the open-book advantage, suggesting the intervention targets pretraining memorization rather than linguistic capability.

---

## Reference

Roi Cohen et al. **Knowledgeless Language Models: Suppressing Parametric Recall for Evidence-Grounded Language Modeling.** arXiv:2607.12831, July 2026. https://arxiv.org/abs/2607.12831
