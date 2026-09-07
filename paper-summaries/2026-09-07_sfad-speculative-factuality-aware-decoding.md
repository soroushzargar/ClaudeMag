# SFAD: Speculative Factuality-Aware Decoding

**arXiv:** 2609.00796  
**Submitted:** September 1, 2026  
**Authors:** Guanqiao Chen, Di Wang, Lijie Hu  
**Affiliation:** MBZUAI and collaborating institutions

---

## Headline Finding

SFAD achieves contextually faithful LLM generation at speculative decoding speeds by training a factuality-specialized draft model via Direct Preference Optimization and deploying an Epistemic Friction detector that steers the draft toward faithful outputs without any added inference latency.

---

## Key Findings (Pyramid Layer 2)

1. **Faithfulness and speed are jointly achievable.** SFAD enforces contextual faithfulness at inference time within the speculative decoding framework, adding zero latency overhead vs. standard speculative decoding while reducing hallucinations by up to 38%.
2. **A context-faithful draft model is the key ingredient.** Training the draft model with fine-grained atomic preference data (ConFide) makes it intrinsically more likely to generate contextually faithful tokens, shifting the speculative decoding acceptance distribution.
3. **Epistemic Friction detects hallucinations without a verifier call.** Distributional tension between the draft and the verifier LLM is used online to flag likely hallucinations before the verifier formally accepts or rejects — enabling proactive steering.
4. **ConFide: atomic perturbations produce better preference data.** Preference datasets constructed with fine-grained atomic perturbations (entity swaps, relation inversions, quantifier changes) yield better DPO-trained draft models than coarser contrastive pairs.

---

## Methodology (Pyramid Layer 3)

### Background: Contextual Faithfulness and Speculative Decoding

**Contextual faithfulness** measures whether a model's output is consistent with a given source document or context. Hallucination — generating plausible but unfaithful text — is the primary failure mode in knowledge-intensive applications (RAG, summarization, QA).

**Speculative decoding** uses a small, fast draft model to propose multiple tokens at once, which a larger verifier LLM then accepts or rejects in parallel. When acceptance rates are high, this achieves substantial speedup with identical output distribution to greedy/sampling decoding from the verifier alone.

### SFAD Components

**Component 1: ConFide Dataset Construction**  
ConFide is a context-faithfulness preference dataset built by:
1. Sampling model outputs for knowledge-intensive prompts (RAG, summarization).
2. Constructing negative examples via atomic perturbations: entity swaps (replacing entity names with plausible alternatives), relation inversions (changing predicate direction), quantifier changes, temporal shifts.
3. Pairing original (faithful) outputs with perturbed (unfaithful) alternatives to create preference pairs.

**Component 2: DPO-Trained Draft Model**  
The draft model is fine-tuned with Direct Preference Optimization on ConFide:
```
L_DPO = -E_{(x,y_+,y_-)} [log σ(β(log π_θ(y_+|x) - log π_ref(y_+|x))
                             - β(log π_θ(y_-|x) - log π_ref(y_-|x)))]
```
where y_+ is the faithful output, y_- is the perturbed unfaithful output, and π_ref is the pre-DPO draft model.

**Component 3: Epistemic Friction**  
During inference, SFAD computes a distributional tension score between the draft and verifier distributions at each proposed token:
```
EF(t) = KL(π_draft(·|x,y_{<t}) || π_verifier(·|x,y_{<t})) · w_specialist(t)
```
where w_specialist(t) is a context-specialist certainty weight derived from the verifier's attention to the source document. High EF(t) signals a likely hallucination; the draft is steered (via temperature/repetition adjustment) before the verifier sees the proposal.

### Inference Procedure
1. Draft model generates k-token proposals.
2. Epistemic Friction scores each proposed token; tokens above an EF threshold trigger draft re-sampling.
3. Verifier LLM processes the (steered) proposals in parallel and accepts/rejects via standard speculative decoding acceptance.
4. The resulting output has the verifier's distribution conditional on faithfulness being enforced by the draft.

---

## Technical Formulation

### Acceptance Criterion (Standard Speculative Decoding)
The token y_t is accepted with probability:
```
p_accept(y_t) = min(1, π_verifier(y_t|x,y_{<t}) / π_draft(y_t|x,y_{<t}))
```

### SFAD Steered Draft Distribution
Under Epistemic Friction, the effective draft distribution becomes:
```
π_steered(y_t) ∝ π_DPO_draft(y_t) · exp(-α · EF_gradient(y_t))
```
where EF_gradient is the token-level gradient of the Epistemic Friction score.

### Faithfulness Gain Bound
The paper proves that under the DPO training setup, the acceptance rate of faithful tokens increases by at least:
```
Δ_accept ≥ β · (E[log π_DPO(y_+)] - E[log π_ref(y_+)]) / Z
```
which is positive whenever DPO training improves the faithful log-probability above the reference model.

---

## Experiments

### Benchmarks
- **RAGTruth:** Factual consistency in retrieval-augmented generation.
- **TruthfulQA:** Truthfulness under adversarial prompts.
- **SummEval:** Faithfulness of document summaries.
- **Metrics:** FactScore, BERTScore-F (faithfulness), tokens/second (throughput).

### Results
| Method                        | FactScore | Throughput (tok/s) |
|------------------------------|-----------|-------------------|
| Greedy (verifier only)        | 61.2%     | 38                |
| Contrastive decoding          | 68.4%     | 19 (2× slower)    |
| Standard speculative decoding | 61.3%     | 91                |
| **SFAD (ours)**               | **72.8%** | **88** (≈ same)   |

SFAD improves FactScore by 11.6 points over greedy and 4.4 points over contrastive decoding, while matching standard speculative decoding throughput.

### Hallucination Reduction
On RAGTruth, SFAD reduces hallucination rate from 28.3% (greedy) to 17.5% — a 38% relative reduction — with no throughput penalty.

---

## Ablations and Interpretation

- **Without ConFide DPO:** Removing the draft model fine-tuning reduces FactScore by 6.8 points, confirming that the draft model is the primary source of faithfulness gain.
- **Without Epistemic Friction:** Removing the online steering reduces FactScore by 2.1 points, showing that EF provides an additional, complementary layer of faithfulness enforcement.
- **Atomic vs. coarse perturbations in ConFide:** Atomic perturbations yield 3.4 points better FactScore than sentence-level contrastive pairs, validating the fine-grained dataset construction.
- **Specialist weight w_specialist:** Using attention-based specialist weighting vs. uniform weighting improves FactScore by 1.8 points — the model is more accurate at detecting hallucinations in high-context-relevance regions.

---

## Reference

Guanqiao Chen, Di Wang, and Lijie Hu. **SFAD: Speculative Factuality-Aware Decoding.** arXiv:2609.00796, September 2026. https://arxiv.org/abs/2609.00796
