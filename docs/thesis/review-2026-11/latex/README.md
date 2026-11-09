# LaTeX source for the literature review

This folder contains a LaTeX conversion of the November 2, 2026 review. It is generated from `../review.md`, the project BibTeX bibliography, and the saved seed-42 development summary. The manuscript wording and evidence cutoff are preserved.

## Files

- `review.tex`: complete article source, tables, equations and TikZ diagrams.
- `references.bib`: bibliography used by the review.
- `assets/pilot-communication.png`: chart generated from the saved development summary.
- `../build_latex.py`: rebuilds this folder from project sources.

## Compile

Use XeLaTeX and Biber (Overleaf: select XeLaTeX, then compile):

```sh
xelatex review.tex
biber review
xelatex review.tex
xelatex review.tex
```

The project WSL image does not currently include a TeX engine, so this environment generated and statically checked the source but did not compile a second PDF. The existing PDF remains the visually reviewed version.
