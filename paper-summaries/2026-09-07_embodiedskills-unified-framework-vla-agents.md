# EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents

**arXiv:** 2609.01281  
**Submitted:** September 1, 2026  
**Authors:** Wei Wang, Wenqiao Zhang, Yutong Lin, Yuqian Yuan, Tianwei Lin, Jinhao Mao, Zhenxuan Fan, Mingjian Gao, Yang Dai, Wentong Li, Zheqi Lv, Zheng Dong, Yingjie Niu, Jiaqi Zhu, Jun Xiao, Chao Li, Yueting Zhuang  
**Affiliation:** Zhejiang University and collaborating institutions

---

## Headline Finding

EmbodiedSkills achieves 86.20% average success across 50 diverse robot manipulation tasks by organizing VLA policy execution around executable skills with prerequisite checking and outcome verification — enabling reliable long-horizon manipulation without retraining the core policy.

---

## Key Findings (Pyramid Layer 2)

1. **Skills-as-execution-proposals enable reliable recovery.** By checking prerequisites before and verifying outcomes after each skill execution, the framework catches failures early and triggers recovery behaviors before errors compound.
2. **Fixed skill interface enables policy modularity.** The shared executable-skill interface separates high-level planning from low-level VLA execution. Low-level VLA policies can be replaced or fine-tuned without changing the agent loop.
3. **Task adaptation outperforms frozen VLA policies.** Task-adapted low-level VLA policies trained within the EmbodiedSkills framework achieve 86.20% success on RoboTwin 2.0 and 97.40% on LIBERO, substantially above frozen policy baselines.
4. **Structured verification reduces cascading failure.** In long-horizon tasks (6+ skills), the verification mechanism reduces cascading failure rates by 62% compared to execution without postcondition checking.

---

## Methodology (Pyramid Layer 3)

### Background: VLA Model Limitations for Long-Horizon Tasks
VLA models map visual observations and language instructions directly to robot actions. While effective for short-horizon tasks, they struggle with:
- **Error accumulation:** Early execution errors propagate and compound over long task horizons.
- **No recovery mechanism:** Standard VLA rollouts have no mechanism to detect failure and retry.
- **Monolithic policy:** The entire skill repertoire is embedded in a single model; adapting for a new task requires retraining the full model.

### EmbodiedSkills Design

**Core Abstraction: Executable Skills**  
A skill s is a tuple:
```
s = (name, precondition, policy, postcondition)
```
where:
- `precondition(obs)` → bool: checks whether the robot is in a state where s can be safely executed.
- `policy(obs, goal)` → action: a low-level VLA policy that generates actions to execute s.
- `postcondition(obs)` → bool: checks whether s was successfully completed.

**Agent Loop**  
At each step the agent:
1. **Selects** a skill s from the high-level planner (a VLM prompted with task description and current observation).
2. **Checks precondition:** if false, triggers a recovery skill or replanning.
3. **Executes** the skill policy until postcondition is met or a timeout is reached.
4. **Verifies postcondition:** if false, re-executes or escalates to replanning.

**Training: Task-Adapted Low-Level Policies**  
Low-level VLA policies are adapted per skill using demonstration data + RL fine-tuning within the skill's execution envelope. The fixed interface means adaptation is modular — adapting one skill's policy does not affect others.

---

## Technical Formulation

### High-Level Skill Planning
The high-level planner is a vision-language model prompted as:
```
Given: task goal G, current observation o_t, skill library S
Select: next skill s_t ∈ S and arguments a_t
```
Skill selection uses chain-of-thought reasoning over the current scene description.

### Precondition and Postcondition Learning
Preconditions and postconditions are learned as binary classifiers:
```
f_pre(obs; θ_pre) → P(precondition satisfied)
f_post(obs; θ_post) → P(postcondition satisfied)
```
trained on annotated demonstrations with positive/negative state labels.

### Recovery Policy
If postcondition fails, the recovery policy is invoked:
```
r_t = RecoveryPolicy(obs_t, s_t, failure_mode)
```
Failure modes are classified by the postcondition checker into known categories (object grasped incorrectly, target position missed, etc.), allowing mode-specific recovery actions.

### Task-Adapted VLA Training
Each skill's low-level policy π_s is trained with a combination of behavioral cloning and PPO:
```
L_s(θ) = L_BC(θ) + λ · L_PPO(θ, R_s)
```
where R_s is a skill-specific dense reward derived from the postcondition classifier.

---

## Experiments

### Benchmarks
- **RoboTwin 2.0:** 50 diverse manipulation tasks including pick-and-place, assembly, tool use, and multi-object rearrangement.
- **LIBERO:** Four task suites (LIBERO-Spatial, -Object, -Goal, -Long).
- **Baseline:** Frozen VLA policy (OpenVLA), task-adapted VLA without skill structure.

### Results
| Method                          | RoboTwin 2.0 (avg) | LIBERO (avg) |
|--------------------------------|-------------------|-------------|
| Frozen VLA (OpenVLA)            | 51.4%             | 78.6%       |
| Task-adapted VLA (no framework) | 71.2%             | 89.1%       |
| **EmbodiedSkills (ours)**       | **86.2%**         | **97.4%**   |

### Long-Horizon Task Analysis
On 8-step tasks (requiring 8+ skills), EmbodiedSkills achieves 74.3% completion vs. 31.2% for the task-adapted baseline without skill structure, demonstrating the importance of verification and recovery.

---

## Ablations and Interpretation

- **Without postcondition verification:** Success drops from 86.2% to 61.4% on RoboTwin 2.0, confirming that verification-driven recovery is the largest single contributor to performance.
- **Without precondition checking:** Success drops to 74.8%, showing that early failure detection prevents compounding errors.
- **Fixed vs. adaptive skill library:** Using a fixed predefined skill library matches a dynamically grown library for the tested tasks, suggesting that 20–30 primitive skills cover the majority of manipulation scenarios.
- **VLM planner size:** Using a 7B-parameter VLM planner vs. a 72B-parameter planner reduces success by 4.1% on RoboTwin 2.0, indicating that the skill structure compensates for planner imperfection.

---

## Reference

Wei Wang, Wenqiao Zhang, Yutong Lin, Yuqian Yuan, Tianwei Lin, Jinhao Mao, Zhenxuan Fan, Mingjian Gao, Yang Dai, Wentong Li, Zheqi Lv, Zheng Dong, Yingjie Niu, Jiaqi Zhu, Jun Xiao, Chao Li, and Yueting Zhuang. **EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents.** arXiv:2609.01281, September 2026. https://arxiv.org/abs/2609.01281
