# Current-stage thesis documentation

**28 October: new supervisor priorities.** Continue adaptive LLZ, independently
justify the distortion budget, prepare a PDF literature review with graphs and
tables before the next meeting, and demonstrate the result-generation process
step by step. See the [feedback and research plan](feedback-2026-10-28.md).
The initial [PDF review and worked example](../../outputs/supervisor-followup-20261028/deliverables/adaptive-llz-literature-and-methods-20261028.pdf)
and [meeting walkthrough](../../outputs/supervisor-followup-20261028/deliverables/meeting-walkthrough.md)
use existing evidence. A justified new budget remains open; 5.9 and 7.0 remain
historical heuristic pilot settings. The exact next-meeting date is unspecified.

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
