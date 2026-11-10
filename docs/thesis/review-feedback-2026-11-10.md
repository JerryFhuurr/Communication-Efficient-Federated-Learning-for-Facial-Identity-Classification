# Literature-review feedback — revision to follow the meeting

The supervisor considers the review a useful starting point, but asks for a clearer argument and deeper comparisons rather than simply more references. The full PDF/LaTeX revision has not been performed during the presentation update.

| Priority | Required revision | Completion evidence |
|---|---|---|
| 1 | Explain why FL needs communication compression; introduce FL before methods; define project scope. Device heterogeneity is excluded. | Revised abstract and introduction connect communication cost to the adaptive LLZ research question without claiming adaptation itself is new. |
| 2 | Define DigiFace, Flower and pilot at first use; explain conditional unbiasedness and project-defined dense code; distinguish tested and proposed work. | Early system/implementation summary and plain-language definitions. |
| 3 | Compare closely with *Rate-Constrained Quantization for Communication-Efficient Federated Learning* and apply the same depth across quantization/adaptive methods. | Compare objective, distortion reference and normalization, coded costs and metadata, assumptions and guarantees, experimental setting and evidence. Do not claim a matched RC-FED experiment exists. |
| 4 | Broaden learned compression beyond the AE example; extend feedback discussion to EF21 and relevant developments. | Primary-source coverage distinguishes genuinely learned FL update codecs from general model coding and offline experiments; assumptions checked against no-EF FedAvg local deltas. |
| 5 | Connect sections into one argument; describe search and selection. | Explain why costs, distortion metrics, shared state and guarantees lead to the research question. Reuse dated search/inclusion/access logs and label the review as narrative, not exhaustive. |
| 6 | Make conclusion critical and quantify adaptation's practical value over fixed and lossless LLZ. | Separate implemented selector correctness from learning reliability; report total communication and computation, identify what evidence is still needed. |
| 7 | Disclose LLM assistance. | Brief factual statement covering ChatGPT/Codex use in discovery, synthesis/drafting, code and artifact preparation; retain documented source/access limits. |

## Primary sources checked for the meeting update

- Hamidi & Bereyhi, author preprint v1: https://arxiv.org/html/2409.06319v1 ; ICASSP 2025 DOI: https://doi.org/10.1109/ICASSP49660.2025.10889213 . The full preprint's Sections 2–4 were inspected on 10 November 2026. The published full text was not compared against the preprint during this update.
- EF21, NeurIPS 2021 primary landing page: https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html . Used only to identify the contractive-compression/error-feedback connection; no new detailed proof audit.

## Presentation boundary

The revised deck clarifies setting and scope, adds the RC-FED comparison, retains the saved decision/code/budget/sweep/confirmation evidence, and makes incremental communication and computation costs explicit. The wider learned-compression review and integrated PDF argument remain revision work. Existing model, codecs, training code and experimental results were not changed.
