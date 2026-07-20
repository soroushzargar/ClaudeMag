---
name: Pipeline toolchain
description: What tools are available for PDF generation and combining in the ClaudeMag environment
type: project
---

pdflatex is installed at /usr/bin/pdflatex (TeX Live 2023/Debian). Always run from the articles/ directory with two passes per file.

pdfunite is NOT available (poppler-utils fails to install due to 404s on noble-updates). Use PyPDF2 (v3.0.1, installed) to combine PDFs:

```python
import PyPDF2
writer = PyPDF2.PdfWriter()
for pdf_path in pdfs:
    reader = PyPDF2.PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)
with open(output, 'wb') as f:
    writer.write(f)
```

pypdf (v6.x) has a cffi import error in this environment — use PyPDF2 instead.

texlive packages installed: texlive-latex-base, texlive-latex-recommended, texlive-fonts-recommended, texlive-latex-extra.

**Why:** pdfunite apt install fails with 404s on the environment's package mirror.
**How to apply:** Always use PyPDF2 for PDF combining steps, never pdfunite.
