# Friday literature review: five bounded task blocks

Prepared 29 October for Friday 30 October 2026. These are task specifications;
no automation or research run has been created by this document.

## Shared instructions for every run

Work in `/home/jerrycao/projects/demo-new` in WSL Ubuntu. The old Windows
`C:/Users/fhuur/Documents/Project/flower-ai/demo-new` folder is a launcher,
not the active project. Read the current outline and paper comparison:

- `docs/thesis/literature-review-outline-2026-11.md`
- `docs/thesis/llz-ieee-version-comparison-2026-10-29.md`

Use primary publications, official proceedings or author preprints. Reuse
previously verified notes; check the precise source passages used for new
claims. Label abstracts, full-paper readings and unresolved claims separately.
Keep private/licensed source PDFs and full extracted text out of Git. Save
concise original notes with source links and section/page references in the
versioned review directory below. Do not reproduce large passages of paper text.

For each method record mechanism, representation transmitted, bias/error
assumptions, feedback/state, FL/local-update setting, communication denominator
and metadata, computational costs, evidence limitations and thesis relevance.
Numerical claims need a source location, experimental setting and denominator.
Do not infer accuracy improvements across incomparable datasets or protocols.

These 40-60 minute estimates are work budgets, not guaranteed agent runtimes or
minimum durations. Finish earlier when the deliverable is complete; do not wait
or broaden the task to fill the slot. If access or time prevents a check, save
partial notes and explicit unresolved items. Do not repeatedly retry a blocked
publisher; use an available author manuscript or record the access gap.

Run blocks in order with no overlap. Read earlier block files where available;
if one is partial or absent, continue independent work and disclose the gap.
No application/codec changes, training, test-set evaluation, broad test runs,
commits/pushes, external messages, PDF authoring or slide creation in these
blocks. The final review PDF is a later writing/rendering task.

Save outputs under `docs/thesis/review-preparation-20261030/`. Each block owns
its numbered file. At completion update `progress.md` with block number,
complete/partial/blocked status, sources checked, unresolved items and handoff.
Do not mark an incomplete or inaccessible-source check complete.

## Blocks and acceptance criteria

| Block | Estimated work | Scope | Output | Complete when |
|---|---:|---|---|---|
| 1 | 45-50 min | Quantization and FL update foundations | `01-quantization.md` | QSGD, signSGD and FedPAQ have attributed comparison rows; gradient vs model delta and coding-format distinctions explained |
| 2 | 50-60 min | Sparsification, error feedback and low rank | `02-sparse-feedback-lowrank.md` | Top-k/Rand-k, memory/EF, DGC and PowerSGD covered with index/factor costs and assumptions; L-Set clarification recorded |
| 3 | 50-60 min | Learned and adaptive compression | `03-learned-adaptive.md` | Chandar AE, AdaQuantFL, DAdaQuant and Isik side information mapped to our LLZ question; decoder/setup costs and claim boundaries explicit |
| 4 | 45-60 min | Recent work and novelty check | `04-recent-work-search.md` | Dated query/source/inclusion log; screen at most 8 relevant 2025-2026 candidates, inspect up to 3 closest; report access gaps and nearest alternatives |
| 5 | 45-60 min | Synthesis and citation audit | `05-synthesis.md` | Integrated algorithm table, qualified research-gap paragraph, citation checks and Monday drafting outline; incomplete upstream evidence remains flagged |

Total active work estimate: 3 h 55 min to 4 h 50 min, plus any scheduling gaps.
The bounded recent search is a narrative-review update, not an exhaustive
systematic review or proof that no competing adaptive LLZ method exists.

## Copyable run prompts

Use each prompt with this shared task specification available in the same
WSL project. Schedule starts separately; this file does not register schedules.

### Block 1

Execute Block 1 of docs/thesis/friday-literature-task-blocks-2026-10-30.md in
/home/jerrycao/projects/demo-new, following its shared instructions. Review
QSGD, signSGD and FedPAQ using our existing bibliography and primary sources.
Explain quantization versus encoding, stochastic-gradient aggregation versus
local model deltas, and unbiasedness/error assumptions. Save the attributed
comparison and limitations in review-preparation-20261030/01-quantization.md
under docs/thesis, then update progress.md. Work budget 45-50 minutes; stop
when complete or save an honest partial result. Do not execute another block.

### Block 2

Execute Block 2 of docs/thesis/friday-literature-task-blocks-2026-10-30.md in
/home/jerrycao/projects/demo-new, following its shared instructions. Review
Top-k/Rand-k, Sparsified SGD with Memory, Error Feedback, DGC and PowerSGD.
Distinguish compressor mechanisms from feedback and training stabilizers;
count coordinate indices or low-rank factors and state the original learning
setting. Preserve the supervisor's coordinate-set meaning of L-Set. Save
02-sparse-feedback-lowrank.md in docs/thesis/review-preparation-20261030/ and
update progress.md. Work budget 50-60 minutes; save unresolved items rather
than broadening scope. Do not execute another block.

### Block 3

Execute Block 3 of docs/thesis/friday-literature-task-blocks-2026-10-30.md in
/home/jerrycao/projects/demo-new, following its shared instructions. Review
Chandar et al.'s autoencoder update compression, AdaQuantFL, DAdaQuant and
Isik et al.'s adaptive side-information compression. Explain what is learned
or adapted, setup/decoder/state costs, objectives and assumptions. Compare
these with the IEEE LLZ paper and our fixed-QSGD adaptive p selector. Label
our AE as offline feasibility and distinguish LLZ-SI from distribution-based
coding. Save 03-learned-adaptive.md in docs/thesis/review-preparation-20261030/
and update progress.md. Work budget 50-60 minutes. Do not execute another block.

### Block 4

Execute Block 4 of docs/thesis/friday-literature-task-blocks-2026-10-30.md in
/home/jerrycao/projects/demo-new, following its shared instructions. Conduct
a bounded primary-source search through 30 October 2026, emphasizing 2025-2026
FL adaptive compression, distortion/error budgets, learned coding and lossy
Lempel-Ziv. Log actual queries, sources, search date and inclusion/exclusion
reasons. Screen at most 8 relevant recent candidates and inspect up to 3
closest works; prioritize precise relevance over filling a count. Record
unavailable papers and uncertainty. Do not infer novelty from a failed search.
Save 04-recent-work-search.md in docs/thesis/review-preparation-20261030/ and
update progress.md. Work budget 45-60 minutes. Do not execute another block.

### Block 5

Execute Block 5 of docs/thesis/friday-literature-task-blocks-2026-10-30.md in
/home/jerrycao/projects/demo-new, following its shared instructions. Read
Blocks 1-4 and existing review notes; preserve any incomplete/access-limited
status. Build a readable main algorithm comparison table and a qualified
research-gap synthesis, separating published results from our DigiFace pilot.
Check cited titles, years/venues, source links, duplicated keys and whether
claims have primary-source support. Update references.bib only for sources
actually verified. Save 05-synthesis.md in docs/thesis/review-preparation-20261030/,
with remaining checks, planned original figures and a Monday drafting handoff;
update progress.md. Work budget 45-60 minutes. Do not create the final PDF,
run training or start the next milestone.

## Scheduling example, not an active schedule

Friday 30 October, Europe/Paris: starts at 09:00, 10:15, 11:30, 13:30 and
14:45 provide room for roughly hour-long blocks and a lunch gap. Confirm a
previous run has finished before another edits the shared review files. A
scheduled local run needs access to the WSL project and its sources; test one
manual block before relying on unattended execution. Under the current chat's
Windows workspace permissions, WSL access has required escalation, so scheduled
access must be verified rather than assumed. Keep the computer on and the
app running. Official scheduling guidance:
https://learn.chatgpt.com/docs/automations
