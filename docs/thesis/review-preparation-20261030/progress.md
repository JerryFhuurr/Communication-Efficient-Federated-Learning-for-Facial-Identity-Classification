# Literature-preparation progress

Last updated **2 November 2026**. Blocks 1–5 were completed on 29 October and
consolidated into the literature-review PDF for sharing by 4 November. The next
supervisor meeting is 10 November. This is a preparation record, not a new
experiment batch.

| Block | Status | Deliverable |
|---|---|---|
| 1: Quantization and FL foundations | Complete | [01-quantization.md](01-quantization.md): QSGD, signSGD, FedPAQ, assumptions, gradient/delta notation and encoding |
| 2: Sparsification, EF and low rank | Complete | [02-sparse-feedback-lowrank.md](02-sparse-feedback-lowrank.md): Top-k/Rand-k, memory/EF, DGC, PowerSGD, costs and L-Set clarification |
| 3: Learned and adaptive compression | Complete | [03-learned-adaptive.md](03-learned-adaptive.md): Chandar AE, AdaQuantFL, DAdaQuant, side information and current LLZ selector |
| 4: Recent-work search | Complete bounded search; R3 full-text follow-up completed 2 Nov | [Recent-work search log](../../../outputs/literature-blocks-20261029/recent-search-log.json): seven screened candidates, three targeted readings, dated R3 verification |
| 5: Synthesis and citation audit | Complete | [05-synthesis.md](05-synthesis.md): integrated comparison, qualified research gap, pilot arithmetic and citation audit |

## Evidence and verification

Fourteen primary PDFs were retrieved with version URLs and SHA256 hashes in
`outputs/literature-blocks-20261029/source-manifest.json`. The bounded search
screened seven 2025–26 candidates and completed three new targeted readings.
On 2 November, the final OpenReview PDF verified the closest discrepancy-aware
candidate as an ICLR 2026 paper; its method and the limits of the comparison are
recorded in the [search log](../../../outputs/literature-blocks-20261029/recent-search-log.json)
and in the review. No candidate count was added by this follow-up.

The QSGD and Top-k byte examples were deterministic codec accounting checks.
They were not training experiments. The 2 November PDF instead uses the saved
WSL seed-42 development batch for QSGD, QSGD+LLZ at p=0 and p=1, and adaptive
LLZ. It labels results as one-seed validation evidence, does not evaluate the
test split, and keeps the training-upload versus total-serialized-object
accounting distinction explicit. The saved protocol, study record and source
manifest are hashed in the build manifest.

## Remaining evidence limits

- ASCEND v1 contains an unresolved proof-report reference; no independent proof validation or paper reproduction was attempted.
- The adaptive LLZ tau budget remains a historical development heuristic. This work does not establish a learning-safe distortion bound or convergence theorem for the local-epoch selector.
- Autoencoder evidence is offline feasibility only. LLZ-SI, Rand-k, DGC and PowerSGD are not implemented in the project comparison, and adaptive LLZ has no error feedback.
- The review is a bounded narrative review, not an exhaustive systematic review or proof of novelty.

## PDF handoff

The refreshed review was rendered and visually checked on 2 November. The
[14-page PDF](../../../outputs/literature-review-20261102/output/pdf/compression-for-federated-learning-review.pdf)
contains the algorithm-family map, literature comparison tables, original
diagrams, a worked LLZ example, current pilot results, 24 references, and
explicit claim/access limitations. See the [editable sources and rebuild notes](../review-2026-11/README.md).
The PDF is ready for user review and sharing by 4 November; the scientific
budget/theory questions and next experiment protocol remain open.
