# Rethinking On-Policy Distillation of Large Language Models II: One Training Example

**arXiv:** 2609.04172  
**Submitted:** September 3, 2026  
**Authors:** Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, Chaojun Xiao  
**Affiliation:** Tsinghua University and collaborating institutions  
**Code:** https://github.com/Thinking-Space/One-Shot-OPD

---

## Headline Finding

On-policy distillation (OPD) of large language models recovers most of the gain achieved by full-dataset training when using a single training query — revealing that OPD is data-overfed but algorithm-starved.

---

## Key Findings (Pyramid Layer 2)

1. **One query suffices for most of the gain.** One-shot OPD improves for hundreds of gradient steps and recovers the majority of full-data OPD's accuracy gain on mathematical reasoning, code, and general reasoning benchmarks.
2. **State coverage explains data efficiency.** A single query already achieves 71.5% state coverage — the fraction of states the full dataset's rollouts visit. Only 16 semantically distinct queries are needed to reach 98.9% state coverage and match full-dataset performance.
3. **Alignment is algorithm-bottlenecked.** Validation accuracy plateaus at a similar pace whether OPD trains on one query or the entire dataset. This reveals that the bottleneck is not data diversity but rather the optimization algorithm itself.
4. **Semantic diversity of queries matters more than quantity.** Adding queries that cover new states (semantically distinct prompts) drives gains; adding redundant queries yields diminishing returns.

---

## Methodology (Pyramid Layer 3)

### On-Policy Distillation Recap
OPD combines student-generated rollouts with dense token-level supervision from a teacher LLM. Concretely:
- The student LLM generates responses to training queries.
- The teacher LLM evaluates and provides per-token supervision signals (e.g., log-probabilities or reward labels).
- The student is updated using these signals, with rollouts updated each iteration.

### Data-Minimal Limit Experiment
The authors hold the number of training steps fixed and vary only the number of unique training queries, from N=1 (single query) up to the full dataset. They measure:
- **Accuracy** on held-out benchmarks (MATH, GSM8K, LiveCodeBench, MMLU-Pro).
- **State coverage:** the fraction of rollout states observed under the full dataset that the reduced query set also visits.

### State Coverage Metric
State coverage is defined as:
```
Coverage(Q) = |S(Q) ∩ S(Q_full)| / |S(Q_full)|
```
where S(Q) is the set of intermediate rollout states reached by queries Q.

---

## Technical Details (Pyramid Layer 4)

### Distillation Objective
The OPD loss is a token-level imitation loss:
```
L_OPD(θ) = -E_{q ~ D} E_{y ~ π_θ(·|q)} Σ_t log π_teacher(y_t | q, y_{<t})
```
where π_θ is the student, π_teacher is the (frozen) teacher, y is a student-generated rollout, and q is the training query.

### Single-Query Surprising Stability
With a single query q*, the gradient at step k is:
```
∇_θ L = -E_{y ~ π_θ(·|q*)} Σ_t ∇_θ log π_θ(y_t | q*, y_{<t}) · log π_teacher(y_t | q*, y_{<t})
```
Despite the fixed input q*, the rollout distribution π_θ(·|q*) shifts at each step, providing gradient diversity via changes in the student's own distribution.

### Coverage Saturation
The coverage curve C(N) as a function of query count N follows an approximate power law: C(N) ≈ 1 - α·N^{-β} for fitted constants α, β. Empirically, C(1) ≈ 0.715 and C(16) ≈ 0.989, showing rapid saturation.

---

## Experiments

### Models and Datasets
- **Student models:** Qwen2.5-7B, Llama-3.1-8B, DeepSeek-R1-Distill-7B.
- **Teacher:** A larger model in the same family (e.g., Qwen2.5-72B or GPT-4o).
- **Benchmarks:** MATH-500, GSM8K, LiveCodeBench, MMLU-Pro.

### Results (MATH-500, Qwen2.5-7B student)
| Query Count | MATH-500 Accuracy | Coverage |
|-------------|------------------|----------|
| 1           | 68.4%            | 71.5%    |
| 4           | 70.1%            | 89.2%    |
| 16          | 71.8%            | 98.9%    |
| Full (512)  | 72.3%            | 100%     |

### Cross-Domain Generalization
The single-query result holds across mathematical reasoning, code generation, and general language understanding, and is consistent across Qwen, Llama, and DeepSeek model families.

---

## Ablations and Interpretation

- **Random vs. semantic query selection:** Randomly sampled queries from the full dataset underperform semantically diverse queries selected to maximize state coverage, confirming that coverage is the active ingredient.
- **Step budget analysis:** One-shot OPD requires more steps to reach the same accuracy as full-data OPD but converges to a comparable final value, suggesting the algorithm can compensate for data sparsity given sufficient compute.
- **Teacher quality:** Experiments with weaker teachers (same-size or smaller) show diminished but still positive one-shot gains, indicating robustness to the teacher-student capacity gap.

---

## Reference

Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, and Chaojun Xiao. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, September 2026. https://arxiv.org/abs/2609.04172
