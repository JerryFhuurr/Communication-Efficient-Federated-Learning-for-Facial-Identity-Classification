# November literature-review PDF

Updated 2 November 2026 for the 4 November sharing target and 10 November
supervisor meeting. This is a focused narrative review. The main search cutoff
is 29 October, with a targeted 2 November follow-up; it is not an exhaustive
systematic review or a convergence proof.

## Deliverable

[Open the updated checked PDF](../../../outputs/literature-review-20261102/output/pdf/compression-for-federated-learning-review.pdf)

The PDF contains 12 main pages and two reference pages: algorithm-family map,
communication definitions, quantization, sparse/feedback and low-rank methods,
learned coding, LLZ-p, adaptive/side-information work, recent-work limitations,
research positioning, and the completed WSL seed-42 DigiFace development results.
It verifies the ICLR 2026 discrepancy-aware alternative and labels all pilot
results as one-seed validation evidence.
There are 24 numbered references, original diagrams, tables and a two-panel
pilot figure. The closest discrepancy-aware alternative and its dated follow-up
have a traceable screening record.

## Editable sources and rebuild

- [review.md](review.md): structured prose, citation keys and figure/table markers.
- [build_review.py](build_review.py): PDF layout, original diagrams, table data,
  frozen-summary plot and input/PDF fingerprints.
- [references.bib](../references.bib): primary-reference metadata and version notes.
- [Preparation notes and citation audit](../review-preparation-20261030/progress.md).

Run inside Ubuntu WSL from the project root:

```bash
cd /home/jerrycao/projects/demo-new
.venv/bin/python docs/thesis/review-2026-11/build_review.py
```

The builder uses ReportLab, matplotlib, NumPy and pypdf from the existing
project environment, plus the system DejaVu fonts. It requires the saved pilot
summary/protocol, AE decision and source manifest at the paths recorded in
`build-manifest.json`; those ignored artifacts must be supplied separately for
a fresh checkout. Dataset images and training are not needed to rebuild.
The pilot table/plot read saved JSON; AE table values were checked against the
saved 6 October decision. The small original LLZ phrase example is generated
and decoded using the independent project codec, without an FL run.

Generated assets, the PDF, and build fingerprints remain under ignored
`outputs/literature-review-20261102/`. The root-level `build-manifest.json` records
input hashes, the PDF hash and page map, and the LLZ worked-example decode checks.
Source publication PDFs are not redistributed. No application/codec behavior or environment was changed.

## Verification and interpretation

All 14 rendered pages were visually inspected; the corrected chart legend and
access note were rechecked in the final rendering. Automated checks found no
page-bound, unresolved citation/template, key numerical-value or input/PDF hash
issues. The final PDF contains 63 hyperlink annotations, including internal references and
primary sources. This is document/source QA, not independent proof validation.

The closest discrepancy-aware paper is now verified as published at ICLR 2026 and is discussed as prior task-sensitive compression.
ASCEND's inspected preprint has an unresolved proof-report reference. A safe
adaptive LLZ budget and a theorem for the actual local-epoch implementation
remain open. The one-seed pilot and offline AE result are labeled accordingly.

The updated PDF is for editorial review and sharing; it has not been sent externally. The next research step is independently specifying the budget
selection/confirmation protocol, while keeping the model and data fixed.

## LaTeX source

A complete LaTeX version is available in `latex/` (`review.tex`, `references.bib`, and the pilot figure), with a project generator in `build_latex.py`. Compile with XeLaTeX and Biber; the WSL environment used to create the source does not have XeLaTeX installed, so the existing visually reviewed PDF remains the compiled copy.
