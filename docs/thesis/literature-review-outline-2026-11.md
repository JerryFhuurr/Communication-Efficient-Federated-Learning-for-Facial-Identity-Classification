# Literature-review outline and work plan, 29 October 2026

Working title: **Compression for Federated Learning: Quantization,
Sparsification, Learned Coding and Adaptive Lempel-Ziv Methods**.

Review PDF target: Wednesday **4 November 2026**. Thursday 5 November is a
fallback requiring communication because the supervisor requested Wednesday.
Meeting: Tuesday **10 November 2026**. Share the five-minute slides and source
code on Monday **9 November in the evening**. No work is scheduled on
31 October-1 November or 7-8 November.

Today's completed source comparison is [IEEE LLZ paper and project mapping](llz-ieee-version-comparison-2026-10-29.md).
This outline is a writing specification, not the completed literature-review PDF.
The existing October PDF is a draft that still needs a complete render/layout
and citation check before being shared.

## 1. Purpose and scope

Produce a focused narrative synthesis suitable for expansion into a thesis
chapter, rather than a list of paper abstracts. Explain the main mechanisms,
what assumptions support them, how communication is measured, and what remains
unresolved for adaptive LLZ. Include work on distributed compression that
informs FL, but label its setting rather than presenting every distributed-SGD
method as a federated local-training result.

Target approximately 10-12 pages of main text plus references. Prioritize a
clear synthesis over reaching a page count. Separate (a) published evidence,
(b) our measured pilot, and (c) prospective experiments/research claims.

## 2. Proposed PDF structure

| Section | Main questions and content | Approximate space | Planned visual |
|---|---|---:|---|
| Abstract and motivation | Why do exchanged bits, local computation and training quality matter together? | 0.5 page | None |
| FL setting and comparison rules | FedAvg/local steps versus gradient aggregation; IID/non-IID; upload/downlink; codec bits versus message/network bytes | 1 page | Original FL communication diagram |
| Quantization | QSGD and sign methods; variance/bias; quantizer versus bitstream format | 1 page | Quantize -> encode -> decode diagram |
| Sparsification and error feedback | Top-k/Rand-k; coordinate-index costs; memory/EF; DGC as a system with stabilizers | 1.5 pages | Residual recurrence and cost illustration |
| Low-rank compression | PowerSGD, matrix structure, rank, aggregation and computation | 0.75 page | Optional matrix-factor sketch |
| Learned compression | Autoencoder update coding; training/decoder distribution cost, drift and generalization | 0.75 page | Included in method taxonomy |
| Source coding, LLZ-p and side information | Reconstructed dictionary, p=0 replay, positive-p bias, LLZ-SI state and metadata | 1.5 pages | Original worked LLZ phrase example |
| Adaptive compression and guarantees | AdaQuantFL, DAdaQuant, side-information methods; what is adapted and what objective is constrained | 1 page | Adaptation comparison table |
| Cross-method synthesis and research gap | Accuracy/bytes/time tradeoff; assumptions and limits; define the narrow adaptive-LLZ question | 1 page | Main algorithm table and evidence-gap table |
| Connection to this thesis | Existing results, limitations and next experiment rationale | 0.5-1 page | Separately labelled project pilot figure |

Keep extended proof derivations, parameter lists and result-generation commands
in an appendix or companion methodology document. The review still needs enough
notation to explain p, QSGD resolution, rho and our separate tau clearly.

## 3. Starting evidence map for the algorithm table

This is an initial representative set, not an exhaustive inventory of current
research. Scoped section-level checks were completed ahead of schedule on 29 October;
see the [five-block synthesis](review-preparation-20261030/05-synthesis.md). Previously written project notes are reading aids, not a
replacement for primary sources.

| Research line | Representative primary sources / bibliography keys | Comparison questions |
|---|---|---|
| FL foundations and measurement | [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html), `mcmahan2017fedavg`; [FL overview](https://arxiv.org/abs/1912.04977), `kairouz2021advances`; field guide `wang2021fieldguide` | What update is sent, how often, and what communication is counted? |
| Quantization | [QSGD](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf), `alistarh2017qsgd`; [signSGD](https://proceedings.mlr.press/v80/bernstein18a.html), `bernstein2018signsgd` | Bias/variance, aggregation rule, payload format and convergence assumptions |
| Sparsification / coordinate sets | [Sparsified SGD with Memory](https://arxiv.org/abs/1809.07599), `stich2018memory`; Top-k and Rand-k | How are coordinates chosen, how are indices encoded, and is residual memory used? |
| Error feedback and practical sparse systems | [Error Feedback](https://proceedings.mlr.press/v97/karimireddy19a.html), `karimireddy2019error`; [DGC](https://arxiv.org/abs/1712.01887), `lin2018dgc` | Separate EF from the compressor; identify momentum/clipping/warm-up contributions |
| Low-rank methods | [PowerSGD](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html), `vogels2019powersgd` | Rank/factor costs, matrix assumptions, extra synchronization and compute |
| Learned encoders | [Chandar et al.](https://arxiv.org/abs/2108.05670), `chandar2021autoencoder` | Prior FL application exists; decoder setup, training cost and update drift need explicit treatment |
| Dictionary / source coding | [LLZ conference](https://doi.org/10.1109/GCWkshps58843.2023.10464660), `zhong2023llz`; [IEEE LLZ article](https://doi.org/10.1109/TCOMM.2026.3745158), `cao2026llz` | Distinguish source coding from quantization, fixed p from adaptive policy, and LLZ-p from LLZ-SI |
| Adaptive quantization | [AdaQuantFL](https://arxiv.org/abs/2102.04487), `jhunjhunwala2021adaquantfl`; [DAdaQuant](https://proceedings.mlr.press/v162/honig22a.html), `honig2022dadaquant` | Which parameter changes across time/clients, and what error/communication constraint is used? |
| Local steps and quantization theory | [FedPAQ](https://proceedings.mlr.press/v108/reisizadeh20a.html), `reisizadeh2020fedpaq` | Which unbiasedness, data and local-update assumptions could or could not transfer? |
| Side-information compression | [Isik et al.](https://proceedings.mlr.press/v238/isik24a.html), `isik2024sideinformation`; LLZ-SI | Distinguish distribution-based stochastic coding from historical-symbol dictionaries |

The supervisor's 'L-Set' means selecting a set of coordinates; use that
clarification to discuss sparsification, rather than assuming a unique named
algorithm. In our implementation Top-k selects largest-magnitude coordinates.

For the final algorithm table, use columns: method/year, research family,
transmitted representation, bias/error assumption, EF/state, data/FL setting,
communication boundary, practical cost/limitation, and relevance to this thesis.
Split across two readable tables if necessary instead of shrinking the font.

### Evidence status as of today

- IEEE LLZ: main text read, relevant pages rendered, earlier version compared.
- AdaQuantFL, DAdaQuant, Chandar AE and Isik side-information: primary landing
  pages/abstracts rechecked on 29 October; existing focused review supplies
  earlier detailed notes. This is not a fresh full-paper audit of those works.
- Other representative works: carried forward from the existing bibliography;
  verify the primary-source passages used in new prose on 30 October.
- A bounded recent-work search through October 2026 remains to be done.
  Do not present this initial set ending at older references as a complete 2026 survey.

## 4. Search and extraction plan for 30 October

Use IEEE Xplore through DTU, primary conference proceedings, arXiv author
preprints and author/institution repositories. Use reviews for orientation,
then cite original sources for algorithmic or numerical claims.

Record search date, query, source, candidate title/year, inclusion reason and
read depth. Starting query groups:

- federated learning compression quantization survey
- federated learning sparsification error feedback Top-k
- federated learning low rank PowerSGD
- federated learning learned compression autoencoder model updates
- federated learning adaptive quantization distortion budget
- federated learning lossy Lempel Ziv adaptive compression

For the recent-work pass, include 2025-2026 and follow citations of the closest
adaptive papers. Screen for an explicit compression mechanism, objective,
learning setting and measurable communication boundary. Keep optimizer-only,
data-compression-only and face-model improvements outside the central synthesis
unless needed to explain a confound. Log exclusions without claiming a formal
systematic-review search or an established novelty result.

For each included paper extract: exact version/venue, mechanism, mathematical
assumptions, update granularity, feedback/state, data heterogeneity, models,
training schedule, communication denominator/metadata, quality-selection rule,
random seeds/uncertainty if reported, computational cost, and limitations.
Missing information remains marked unreported until checked.

## 5. Figures and tables specification

1. Original taxonomy: quantization, sparsification, low rank, learned coding,
   source coding and adaptation as potentially composable operations. EF is a
   cross-cutting state mechanism, not a mutually exclusive compressor family.
2. Original LLZ/adaptive pipeline: local update -> single QSGD draw -> p=0/1/2
   candidates -> local decode -> error/complete-byte measurement -> feasible
   minimum-byte choice. Distinguish original-update, QSGD and LLZ reconstructions.
3. Main algorithm comparison table and a smaller assumptions/evidence table.
4. A separate project-pilot figure from frozen results, labelled DigiFace,
   15 rounds, seed 42, validation-loss checkpoint selection and logical bytes.
   Show uploads and total bytes separately; keep 5.9/7.0 as historical heuristics.

Do not draw a numerical accuracy leaderboard across unrelated papers' datasets,
models and communication denominators. Redraw conceptual diagrams with citations;
any reused publisher figure needs its reuse conditions checked. Every plotted
number must have a source row or saved-result path.

## 6. Provisional synthesis and research gap

Adaptive quantization and side-information FL compression already exist.
The IEEE LLZ article explicitly leaves adaptive dictionary/distortion selection
as future work. A defensible working contribution is therefore narrower:
LLZ-specific client/round selection, full message accounting, computational
cost, and an independently justified/evaluated distortion-budget rule.
The [bounded recent search](review-preparation-20261030/04-recent-work-search.md)
is now complete, with one access-limited source; it does not establish novelty.

Our existing selector uses one p for a whole client update after one QSGD draw,
and chooses the smallest complete reply subject to
`sum ||z_p-z||^2 <= tau * sum ||u||^2`. A reconstruction guarantee is not an
accuracy or convergence guarantee. The review must explain that the old
5.9/7.0 values came from a heuristic calibration, not from the IEEE theorem.
The held-out test split remains reserved. No new tau or acceptance tolerance is
frozen by this outline.

## 7. Revised weekday work plan

The original drafting days were completed early on 29 October. The current
[30 October-9 November plan](plan-2026-10-30-to-11-09.md) supersedes the old
experimental timetable: budget protocol on 30 October, calibration/policy on
2 November, seed-42 development on 3 November, PDF sharing and policy freeze on
4 November, confirmation on 5 November, analysis/demo on 6 November, and the
five-minute meeting package on 9 November. Both weekends are excluded.

The core default is seven matched runs with a bounded development revision;
use the linked plan's compute, provenance and confirmation rules. Do not treat
historical Windows results as matched Linux controls. Budget justification and
quality preservation remain open; the checked PDF is ready for user review and
sharing. Slides/source are due for user sharing on 9 November evening.

## 8. Release checklist for the 4 November PDF

- All main research families represented with primary citations and limitations.
- IEEE LLZ cited as Early Access; previous audit source version preserved.
- Quantization loss, extra LLZ distortion and final learning accuracy distinguished.
- p, quantization levels, tau and rho have explicit units/denominators.
- Published findings, our results and proposed work visibly separated.
- Table IV's PowerSGD setting and Table V's feedback distinction stated correctly.
- Graphs/tables readable after rendering; bibliography resolves; links checked.
- Claims about novelty, statistical equivalence and convergence stay within evidence.
- PDF is a share-ready literature review, not the unreviewed October draft.
