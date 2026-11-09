# RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines

**arXiv:** 2611.08308  
**Authors:** Roko Parac, Christos Ziakas, Leo Ardon  
**Venue:** cs.AI  
**Submitted:** 6 November 2026  
**URL:** https://arxiv.org/abs/2611.08308

---

## Summary

Specifying reward functions for reinforcement learning agents typically requires either hand-engineering a scalar function or providing a fixed VLM-based evaluator that scores image observations against a natural-language goal. Both approaches implicitly assume that reward depends only on the current state (the Markov property). Many real-world tasks violate this: a robot must open a drawer *before* picking up an object, or a navigation agent must visit locations in a specific order. RM-VLM addresses non-Markovian task structure by automatically constructing a reward machine — a finite automaton whose states track task progress — from a natural-language description, while using a vision-language model to ground each automaton proposition in raw image observations.

---

## Method

**Reward machine construction.** Given a natural-language task description T, an LLM decomposes T into a set of propositions {p_1, ..., p_n} (atomic facts that can be true or false in the world) and a finite automaton M = (Q, q_0, q_f, δ, ρ) where:
- Q: states tracking task progress
- q_0: initial state
- q_f: terminal (goal) state
- δ: Q × 2^P → Q: transition function over proposition sets
- ρ: Q × Q → ℝ: reward function over transitions

The LLM generates this structure as structured JSON, which is validated and compiled to a runnable automaton.

**VLM grounding.** At each environment step, a VLM receives the current image observation and each proposition p_i and returns a probability P(p_i = true | image). The resulting binary vector advances M's state through δ, triggering the appropriate reward ρ.

**Dense reward shaping.** To alleviate sparse reward in long-horizon tasks, a VLM-based value estimator V_VLM scores the agent's visual observation against each pending proposition and provides a shaped bonus proportional to estimated transition progress.

---

## Experimental Setup

Three task domains:
1. **Grid-world navigation:** reach goal after visiting a colored marker.
2. **Robotic manipulation (simulated):** place objects in order-dependent sequence.
3. **3D object collection:** collect items in a specified order in a room.

Baselines: VLM-RM (VLM scalar reward), CLIP-reward, TAMER with VLM feedback.  
Metric: task success rate at convergence.

---

## Results

RM-VLM consistently outperforms all baselines in task success across all three domains. The gains are largest on tasks with long dependence chains (manipulation sequence), where flat VLM reward fails to distinguish correct-order from incorrect-order completions.

---

## Key Insight

The key insight is that the structure of a task description is not just semantic content to embed — it encodes a causal, ordered dependency graph that maps directly onto a reward machine's transition structure. By making this structure explicit via an LLM-constructed automaton rather than absorbing it implicitly into a dense embedding, RM-VLM enables an agent to receive precise, non-Markovian credit assignment without any additional human annotation.

---

## Limitations

- LLM-constructed reward machines may contain errors (missing transitions, wrong propositions) that are only discovered at training time; no automated repair mechanism is described.
- The VLM grounding accuracy limits the ceiling performance: if the VLM misclassifies propositions, the automaton's state tracker diverges from the true task state.
- Evaluated only in simulation; real-world deployment with imperfect perception is not tested.
