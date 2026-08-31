---
name: Pipeline toolchain
description: What tools are available for PDF generation and combining in the ClaudeMag environment
type: project
---

pdflatex is installed at /usr/bin/pdflatex (TeX Live 2023/Debian). Always run from the articles/ directory with two passes per file.

pdfunite (poppler-utils) fails to install due to 404s on noble-updates. Use ghostscript (gs) to combine PDFs:

```bash
gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
  -sOutputFile=output.pdf \
  input1.pdf input2.pdf input3.pdf ...
```

ghostscript 10.02.1 is available at /usr/bin/gs (installed via apt --fix-missing). Confirmed working on 2026-08-31.

PyPDF2 v3.0.1 also installed as fallback. pypdf v6.x has cffi import errors — use PyPDF2 if gs unavailable.

texlive packages installed: texlive-latex-base, texlive-latex-recommended, texlive-fonts-recommended, texlive-latex-extra.

**Why:** pdfunite apt install fails with 404s; ghostscript installs cleanly with --fix-missing.
**How to apply:** Use ghostscript (gs) for all PDF combining. Fall back to PyPDF2 only if gs not available.
