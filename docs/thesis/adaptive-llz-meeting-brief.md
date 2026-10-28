# Literature briefing for the next supervisor meeting

Updated 22 October 2026. The agreed adaptive milestone is complete, including
the initial controlled pilot. The exact next-meeting date remains unspecified.
The historical 20 October meeting pack is preserved. Use the new
[meeting pack](../../outputs/meeting-next-20261022/README.md).

## Closest literature and relationship to the project

| Source | Relevant contribution | Distinction for our work |
|---|---|---|
| [QSGD, 2017](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf) | Ideal stochastic quantization with unbiasedness and moment control | Our fixed first stage uses a different dense packet representation. |
| Supplied Cao et al. LLZ revision, [audit](llz-paper-audit.md) | Dictionary coding with signed-symbol tolerance | Our second stage. Symbol error is distinct from recognition error. This source is a manuscript. |
| [AdaQuantFL, 2021](https://arxiv.org/abs/2102.04487) | Quantization precision changes during training | Prior adaptive quantization, rather than LLZ selection after a fixed draw. |
| [DAdaQuant, 2022](https://proceedings.mlr.press/v162/honig22a.html) | Precision varies across time and clients | Closest adaptation precedent. Its allocation model differs from measured LLZ extra error. |
| [FedPAQ, 2020](https://proceedings.mlr.press/v108/reisizadeh20a.html) | Analyses quantized local updates | Relevant theory assumptions. Selected positive-p LLZ can introduce bias. |
| [Error feedback, 2019](https://proceedings.mlr.press/v97/karimireddy19a.html) | Residual correction for biased compression | Context for historical Top-k+EF. The adaptive pilot has no feedback. |
| [Learned update compression, 2021](https://arxiv.org/abs/2108.05670) | Autoencoder compression has an FL precedent | Our AE candidate has offline feasibility evidence only, with no FL accuracy result. |

## Oral literature explanation, about 90 seconds

“I reviewed the closest papers to position the adaptive method properly.
QSGD is our first stage, and the supplied LLZ manuscript defines the second
stage. The lossless setting preserves the quantized update, while positive
tolerance can introduce extra error and bias.

Adaptive quantization already exists. AdaQuantFL varies precision during
training, and DAdaQuant adapts it across both time and clients. Our narrower
question is how to select LLZ tolerance after keeping the QSGD draw fixed,
using the actual message size and a measured error limit.

The pilot now gives us an initial answer. The tighter policy used about
35 percent fewer upload bytes than lossless LLZ, at 83.2 versus 83.4 percent
validation accuracy. The looser policy did not improve on fixed p equals one.
The tighter policy also cost much more computation, so adaptation has a clear
tradeoff in this prototype. These are results from one seed.

For theory, FedPAQ is relevant because it analyses local updates, but its
assumptions do not automatically cover LLZ bias. I derived aggregation and
bias bounds, then a conditional stationarity result. The remaining bridge is
to justify the local-update assumptions for our actual training procedure.
I would like to agree whether that proof or reducing selection overhead
should be the next priority.”

## Interpretation and evidence boundary

- Exact p=0 replay verifies secondary coding relative to QSGD, not recovery
  of the original full-precision update.
- Tight adaptation gives a descriptive accuracy/upload tradeoff. It does not
  establish equivalent accuracy, optimal budgets or statistical superiority.
- Local message-size minimization does not optimize the future trajectory.
  Looser adaptation did not beat fixed p=1 in this pilot.
- Total serialized saving versus p=0 is only 0.514%, and successful-launch
  elapsed time rises from 32.3 to 79.1 minutes for the tighter policy.
- The conditional proof needs local-alignment and moment assumptions that
  have not been established for the actual shuffled-local-epoch setup.

## Priority supervisor questions

1. Should upload bytes at comparable accuracy lead the thesis, or should
   total communication and computation have equal weight?
2. Should the next work reduce candidate-selection overhead or derive
   local-drift conditions for the conditional theorem?
3. Is a one-step stationarity result a useful initial theoretical target
   before the local-epoch extension?
4. Is additional LLZ distortion the right primary constraint, while total
   reconstruction error remains a separate diagnostic?

The [focused review](adaptive-llz-literature-review.md) and
[assumptions map](adaptive-llz-theory-outline.md) give source details.
The [conditional derivations](adaptive-llz-conditional-stationarity.md) state
what has been proved and what remains open. The
[pilot report](../../outputs/adaptive-study-20261021T104244874038Z/report.md)
contains the audited measurements. Any new seed, schedule, feedback policy or
dataset requires a separately agreed experiment batch.
