---
name: ML News Pipeline Setup
description: How the full news production pipeline is structured and executed in this project
type: project
---

The pipeline runs fully within the news-piece-creator agent using WebSearch/WebFetch to fetch papers, then directly writing LaTeX files and compiling with pdflatex, and combining with pdfunite (available at /opt/homebrew/bin/pdfunite).

Sub-agents (ml-paper-fetcher, ml-paper-pyramid, news-pdf-generator, pdf-combiner) are defined in .claude/agents/ as local agent definitions — they are not registered as RemoteTriggers and run as inline logic within the orchestrating agent.

**Why:** RemoteTrigger list is empty; all execution happens within the conversation context.

**How to apply:** On future runs, execute all pipeline steps directly. Do not attempt to launch sub-agents via RemoteTrigger or Skill tool — embody each agent's logic inline.

pypdf and PyPDF2 are not installed. pdfunite (poppler) is the PDF merging tool to use.
pdflatex is available at system PATH (TeXLive 2023). In cloud/linux sessions it may need to be installed first: apt-get install -y texlive texlive-latex-extra.
Output files go directly to /home/user/ClaudeMag/ (Linux) or /Users/soroushzargarbashi/Work-Data/Projects/ClaudeMag/ (macOS).
Paper summaries go to paper-summaries/ subdirectory.
Papers list saved as papers-list-YYYY-MM-DD.txt in the project root.
Final combined PDF named YYYY-MM-DD_ml_news_edition.pdf at the project root (not in editions/).
arxiv.org is not directly accessible via WebFetch in cloud sessions — use WebSearch with specific arXiv IDs and titles, plus secondary sources (GitHub digests, NeurIPS/ICML virtual pages, Semantic Scholar) to gather paper details.
