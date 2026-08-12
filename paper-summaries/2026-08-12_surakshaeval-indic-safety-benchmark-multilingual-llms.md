# SurakshaEval: An Indic Safety Benchmark for Multilingual LLMs

**arXiv:** 2608.07862
**Authors:** Debopriyo Banerjee, Kapil Rajesh Kavitha, Angana Borah, Xudong Han, Yuxia Wang, Parameswari Krishnamurthy, Utkarsh Agarwal, Atharva Kulkarni, Swaran Lata, Ayush Munot, Dhruv Sahnan, Aaryamonvikram Singh, Preslav Nakov, Monojit Choudhury
**Venue:** arXiv preprint
**Date:** August 8, 2026
**URL:** https://arxiv.org/abs/2608.07862

---

## Summary

Existing LLM safety benchmarks almost exclusively target English and Western cultural contexts, leaving the safety behavior of multilingual LLMs in non-Western languages poorly understood. SurakshaEval is a human-written safety benchmark covering ten major Indian languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu) plus English. It includes both generic safety prompts and region- and language-specific prompts capturing localized sociocultural sensitivities. Experiments on leading multilingual LLMs reveal consistent failure modes: over-refusal, missed implicit bias, and insufficient cultural contextual awareness.

## Problem

India has 22 scheduled languages and over 1.4 billion speakers, with growing access to LLM-based systems through multilingual interfaces. Current LLM safety research has almost no coverage of:
- Non-English language safety
- Culturally specific harm categories (communal tensions, caste-based discrimination, regional political sensitivities)
- Implicit biases embedded in Indian language use

English-centric safety training may cause models to refuse harmless queries in Indian languages (over-refusal) while failing to detect genuinely harmful content that exploits cultural context unfamiliar to the training distribution.

## Benchmark Design

SurakshaEval prompts are **human-written by native speakers** in each language, spanning two categories:
1. **Generic prompts:** Safety scenarios common across Indian contexts (violence, self-harm, misinformation)
2. **Culturally specific prompts:** Scenarios that exploit or reference Indian sociocultural context (communal tensions, caste-based language, regional political incitement, religious sensitivities)

The ten languages covered:
- Indo-Aryan: Assamese, Bengali, Gujarati, Hindi, Marathi, Punjabi
- Dravidian: Kannada, Malayalam, Tamil, Telugu

## Key Findings

Three recurring failure modes were identified across all tested LLMs:

1. **Over-refusal:** Models refuse harmless queries in Indian languages, apparently because the language itself triggers safety filters trained primarily on English toxicity patterns
2. **Missed implicit bias:** Models fail to detect harmful content encoded in culturally specific language — slurs, dog whistles, or communal framing that requires cultural knowledge to recognize as harmful
3. **Insufficient contextual awareness:** Models do not modulate responses appropriately for regionally sensitive topics, either ignoring the sensitivity or applying Western-centric framing

## Why It Matters

With over a billion potential users in a single country, language-specific safety gaps represent a major deployment risk. SurakshaEval provides the first systematic evaluation infrastructure for this population. The benchmark's findings — that current multilingual LLMs exhibit systematic over-refusal for some languages while missing culturally embedded harms in others — should inform safety training for any system deployed to Indian-language speakers. The benchmark is reusable and designed to grow as new LLMs are released.

## Key Contributions

1. First human-written, culturally grounded safety benchmark for 10 Indian languages + English
2. Two-tier prompt design: generic + culturally specific
3. Baseline safety evaluation of leading multilingual LLMs across 11 languages
4. Identification of three systematic failure modes: over-refusal, missed implicit bias, cultural insensitivity
