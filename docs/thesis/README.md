# Thesis preparation: literature and experimental design

Prepared 25 September 2026 for the Monday 28 September milestone.

| Deliverable | Document | Status |
|---|---|---|
| Focused literature synthesis and research gap | [Literature review](literature-review.md) | Written; primary references checked |
| Research questions, controls, metrics, and staged experiments | [Experimental design v1.0](experimental-design.md) | Written; current results and configuration reconciled |
| Reusable citations | [BibTeX bibliography](references.bib) | Includes the supplied revision explicitly as a manuscript |

Read the literature review first, then the protocol. This completes the two
preparation documents for supervisor discussion; it does not imply that the
thesis literature chapter, algorithm audit, or future experiments are finished.
The review is a focused synthesis, not an exhaustive systematic review.

The documents distinguish published findings, a supplied revision, completed
local experiments, and proposed work. No new training or test-set evaluation
was performed to prepare them. Existing uncommitted implementation changes were
left intact. Source PDFs remain in their original location; extracted text,
page previews, and the evidence index are ignored under
`outputs/monday-preparation/`. The written synthesis and protocol belong in Git.

## Decisions for the supervisor

1. Is the principal contribution an empirical study of LLZ-p on FedAvg model
   updates, or is a new compression algorithm expected? The current results
   support the former; they do not establish algorithmic novelty.
2. Is closed-set synthetic identity classification an acceptable demonstration,
   and which additional dataset or task is necessary for the final claims?
3. Should the later lossy study prioritize error feedback, given the revised
   paper's experimental setup, before LLZ-SI or other extensions?
4. Are the proposed paired seeds, descriptive statistics, and 1.0 percentage-point
   practical accuracy margin appropriate? This margin is a prospective decision
   aid, not a retrospectively preregistered hypothesis test.

Next week's implementation milestone remains the paper-to-code audit, any
required corrections, timing instrumentation, focused validation, and release
documentation. This preparation does not launch that work automatically.
