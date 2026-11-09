# Current-stage thesis documentation

**Revised pre-meeting schedule:** the [30 October-9 November plan](plan-2026-10-30-to-11-09.md)
uses the completed PDF to bring budget justification, policy implementation,
matched development/confirmation runs and the demonstration forward. Literature
sharing remains 4 November; slides/source sharing remains 9 November evening.

**2 November: literature-review PDF complete.** The
[14-page review](../../outputs/literature-review-20261102/output/pdf/compression-for-federated-learning-review.pdf)
contains 24 references, original diagrams/tables, the seed-42 pilot graph and
explicit evidence boundaries. All pages were visually inspected; extracted-text
checks confirmed the key results, 24 references and 63 hyperlink annotations.
The final OpenReview PDF for the closest discrepancy-aware alternative was
verified as ICLR 2026. [Editable sources and rebuild instructions](review-2026-11/README.md)
are retained. The PDF is ready for review/sharing by 4 November.

**29 October: plan for the 10 November meeting.** The literature-review PDF
is due for sharing on **4 November**; slides and source code should be shared
on **9 November in the evening**. Start with the
[review outline and weekday work plan](literature-review-outline-2026-11.md)
and [IEEE paper version comparison and project mapping](llz-ieee-version-comparison-2026-10-29.md).
The 21 October IEEE Early Access publication has now been verified from the
publisher PDF. Earlier manuscript-source records remain historical.
The existing 28 October PDF is a draft awaiting full citation and render/layout
review; it is not the final 4 November deliverable. No new training was run
for today's source-reading and outline task.

**29 October preparation blocks complete; 2 November source follow-up added.**
All five [preparation blocks](review-preparation-20261030/progress.md) include
source notes, a bounded recent search and an [integrated synthesis](review-preparation-20261030/05-synthesis.md).
Bibliography metadata, local links, source fingerprints and pilot arithmetic
were checked. The closest discrepancy-aware source was verified from its final
OpenReview PDF on 2 November; abstract-only candidates remain identified as
such. ASCEND's proof-report reference remains open. The adaptive-budget audit
is summarized below. The current PDF deliverable is linked above.

**28 October: new supervisor priorities.** Continue adaptive LLZ, independently
justify the distortion budget, prepare a PDF literature review with graphs and
tables before the next meeting, and demonstrate the result-generation process
step by step. See the [feedback and research plan](feedback-2026-10-28.md).
The initial [PDF review and worked example](../../outputs/supervisor-followup-20261028/deliverables/adaptive-llz-literature-and-methods-20261028.pdf)
and [meeting walkthrough](../../outputs/supervisor-followup-20261028/deliverables/meeting-walkthrough.md)
use existing evidence. A matched single-seed sweep provisionally supports the
budget choice: 5.9 is the only tested value that passes the frozen gate, not a
proven accuracy-safe constant. The
[3 November sensitivity audit](adaptive-budget-sensitivity-2026-11-03.md)
records their exact origin, frozen-update replay, and a matched seed-42 sweep of
tau=5.9, 6.5 and 7.0. Only 5.9 passes the predeclared 1-point validation
gate; the report includes a full-run accuracy/upload/distortion plot, row-level
selection audits and reproduction steps. The 5.9 choice remains provisional
until confirmation on additional seeds.
The meeting is 10 November; see the current plan above.

**22 October: agreed next-meeting milestone complete.** The
[next-meeting pack](../../outputs/meeting-next-20261022/README.md) includes an
editable PowerPoint with five main slides and three optional appendices,
a three-minute script, extended speaking notes and supervisor questions.
The [literature briefing](adaptive-llz-meeting-brief.md) now incorporates the
completed adaptive pilot. The [conditional stationarity derivations](adaptive-llz-conditional-stationarity.md)
extend the earlier bounds under stated alignment and moment assumptions.
Those assumptions have not been justified for the actual shuffled local epochs,
so a full theorem for that implementation remains open. No new experiment was
launched during this meeting/documentation completion.

**Historical 20 October update:** the next research direction is adaptive LLZ distortion,
following the supervisor's notes. Start with the
[focused adaptive-LLZ literature review](adaptive-llz-literature-review.md) and
[short next-meeting literature briefing](adaptive-llz-meeting-brief.md).
The [21–27 October work plan](plan-2026-10-21-to-27.md) prepares the following
meeting over five working days, excluding the weekend and 28 October.
The earlier five-method comparison (including uncompressed and Top-k+EF) is
complete for seeds 42/43/44, and the AE has
an offline feasibility result without FL integration. These updates supersede
the September future-work status below. The
[adaptive selector implementation contract](adaptive-llz-contract.md) records
the 21–22 October work brought forward to 20 October. Its matched adaptive
pilot is now complete; a full convergence theorem remains open.

**21 October:** the [adaptive experiment protocol](adaptive-experiment-protocol.md)
specifies the saved-update calibration and five-method pilot. The
[literature assumption map and theory outline](adaptive-llz-theory-outline.md)
provides the aggregation-error bound, a conditional composition/bias bound and
a one-round descent inequality; a full convergence theorem remains open.

The [completed local pilot report](../../outputs/adaptive-study-20261021T104244874038Z/report.md)
contains the five new seed-42, 15-round runs. Fixed p=0 exactly replayed QSGD
(83.4% selected validation accuracy). The tighter policy reached 83.2% with
35.45% fewer upload bytes than p=0; the looser policy reached 78.5% and did
not improve on fixed p=1's accuracy/upload point. Calibration, plots, timing
and the single-seed/measurement limitations accompany the report. These are
distinct from the earlier three-seed study.

## Historical September documentation

The following section records the 25 September package for the 28 September
meeting. Its future-work list describes that earlier stage. Use the dated
October updates above for current completion status.

This package brings together the literature, implemented methodology, experimental
design, and interpretation of completed results. It completes the writing for
the current stage; future experiments and their final conclusions remain separate work.

## Reading order

| Document | Purpose |
|---|---|
| [Monday meeting guide](meeting-2026-09-28.md) | Five-minute progress update, key results, and questions about future work |
| [Supervisor summary](supervisor-summary.md) | Short progress briefing, main findings, and questions requiring direction |
| [Literature review](literature-review.md) | FL communication costs, QSGD, LLZ-p, alternatives, and the research gap |
| [Methodology](methodology.md) | Implemented data, model, federated update, codec, measurement, and reproducibility workflow |
| [Experimental design v1.0](experimental-design.md) | Research questions, controls, metrics, provenance, and bounded follow-up plan |
| [LLZ-p paper-to-code audit](llz-paper-audit.md) | Algorithm interpretation, implementation mapping, limitations, and correctness evidence |
| [Results and discussion](results-discussion.md) | Matched pilot results, historical context, threats to validity, and conclusions |
| [BibTeX bibliography](references.bib) | References, including adaptive compression and the supplied revision explicitly as a manuscript |

Start with the supervisor summary for a meeting, or read the literature review
and methodology for the technical narrative. The review is a focused synthesis,
not an exhaustive systematic review. Published findings, manuscript claims,
local measurements, and proposed work are distinguished throughout.

## Completion and evidence

The standalone codecs, federated integration, matched seed-42 pilot, and bounded
paper-to-code audit are completed. The audit found no blocking codec defect
under its documented interpretation and passed 72 focused tests. It does not
constitute author certification or a convergence proof.

Dedicated codec timing, stronger-model seed confirmation, error feedback,
Top-k, non-IID experiments, and LLZ-SI are future work. The protocol defines
the next proposed batch; this documentation does not launch it automatically.
Supervisor decisions are collected in the [briefing](supervisor-summary.md#decisions-requested).

No new training, test-set evaluation, or implementation tests were run during
this consolidation. Existing evidence was reused; local links and result
arithmetic were checked. Written Markdown and BibTeX belong in Git. Source PDFs
remain in their original locations. Results, extracted paper text, and audit
records remain excluded from Git under `outputs/`, including:

- `outputs/monday-preparation/`: source extraction and preparation evidence.
- `outputs/llz-paper-audit/`: audit fingerprints and focused test record.
- `outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/`:
  primary matched pilot, source snapshot, checkpoints, and plots.
- `outputs/thesis-documentation/`: documentation consistency check.

Links to ignored artifacts work in this workspace but require those artifacts
to be supplied separately with a fresh checkout. The methodology and protocol
explain the dataset, cache, weights, and initialization required for reproduction.
