# Adaptive LLZ: literature assumptions and theoretical starting point

Prepared 21 October 2026 for the work originally planned for 23 and 26 October.
The inequalities below are elementary derivations for a stated mathematical
model. They are not a completed FedAvg convergence theorem or an accuracy guarantee.
The implementation contract remains [adaptive-llz-contract.md](adaptive-llz-contract.md).

## Literature matrix and proof obligations

| Reference | What changes / what is established | Assumption or limitation relevant here |
|---|---|---|
| [QSGD, Alistarh et al., 2017](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf) | Stochastic quantization with conditional unbiasedness and a norm-relative second-moment bound in its ideal model | Our fixed-width packet format is different from its coding analysis. Positive-p LLZ is a second, potentially biased stage. |
| Supplied Cao et al. LLZ revision, audited in [paper-to-code mapping](llz-paper-audit.md) | Signed-symbol tolerance after quantization; reconstructed dictionary; compressor-level distortion control | p bounds integer-code error, not recognition accuracy. Its stated gradient analysis is not automatically our multiple-local-step FedAvg analysis. |
| [AdaQuantFL, Jhunjhunwala et al., 2021](https://arxiv.org/pdf/2102.04487), Sections 3–4 and Appendix C | Adapts quantization resolution over time and examines progress against communicated bits | Its analysis uses unbiased quantization with controlled variance, smoothness and stochastic-gradient assumptions. It does not establish an LLZ selection-bias bound. |
| [DAdaQuant, Hönig et al., 2022](https://proceedings.mlr.press/v162/honig22a/honig22a.pdf), Section 3.5 / Theorem 1 | Allocates client quantization levels under a quantization-variance constraint; adapts over clients and time | The allocation result uses its particular stochastic fixed-point/uniform-parameter model. Our criterion is a measured per-update error bound after fixed QSGD. |
| [FedPAQ, Reisizadeh et al., 2020](https://proceedings.mlr.press/v108/reisizadeh20a/reisizadeh20a.pdf), Assumptions 1–3 / Section 4 | Analyses quantized local updates with periodic averaging and partial participation | Unbiased compression, smoothness and controlled gradient noise are explicit; Assumption 4 additionally concerns its strongly convex case. Its common-distribution setup does not prove our finite empirical objectives identical. |
| [Error feedback, Karimireddy et al., 2019](https://proceedings.mlr.press/v97/karimireddy19a.html) | Addresses biased compression using a residual mechanism under its analysed conditions | The current five-method QSGD/LLZ batch has no feedback. A theorem for a residual algorithm cannot be imported without that recursion. |

The adaptive-paper full texts and official records were revisited on 21 October.
The broader [focused literature review](adaptive-llz-literature-review.md) covers
sparsification, learned compression and side information. No claim of priority
for adaptive FL compression follows from this review. The narrow research
question is selection of the *second-stage* LLZ tolerance under actual-byte and
measured-error constraints, followed by an analysis matching that selection rule.

## 1. A deterministic bound for the extra LLZ stage

Fix a communication round and its actual local updates. Let client weights
`a_i >= 0` sum to one; the implementation uses sample counts. Write:

```
u_i = local model delta before compression
z_i = decoded QSGD reconstruction
e_i = decoded selected LLZ reconstruction - z_i
tau_i = configured relative squared-error budget
```

The client checks `||e_i||² <= tau_i ||u_i||²`, summing over all transmitted
tensors. A zero update requires zero extra error. In real arithmetic the
aggregate extra error `e_bar = sum_i a_i e_i` satisfies

```
||e_bar||² <= sum_i a_i ||e_i||²
            <= sum_i a_i tau_i ||u_i||².                  (1)
```

The first inequality is convexity of squared norm. It needs neither independent
client errors nor zero-mean LLZ error. Choosing the smallest feasible packet
after inspecting the update does not invalidate (1): every possible selected
candidate must satisfy the constraint. It also does not make that error unbiased.

This is the proved compressor/aggregation statement. It concerns one round's
already-computed local updates, not the loss after many rounds. Torch aggregation
has additional finite-precision rounding; (1) describes the ideal real-arithmetic
model rather than a bitwise floating-point bound on aggregation.

## 2. Conditional composition with ideal QSGD

Condition on a fixed local update u. Let `q = z-u`. Suppose the first-stage
quantizer satisfies `E[q | u]=0` and `E[||q||² | u] <= omega ||u||²`. These are
mathematical assumptions, not exact floating-point unbiasedness claims.
The selected LLZ error may depend on the same QSGD draw. Nevertheless,
`E[||e||² | u] <= tau ||u||²`, since its constraint holds for each draw.
The L2 triangle inequality for random vectors gives

```
E[||q+e||² | u] <= (sqrt(omega)+sqrt(tau))² ||u||².       (2)
||E[q+e | u]|| <= sqrt(tau) ||u||.                       (3)
```

Equation (3) uses ideal QSGD unbiasedness and Jensen's inequality. It bounds
possible bias; it does not assert the bias attains that bound. No independence
between q and e was assumed. The cross term prevents simply adding omega and tau.

For per-tensor QSGD one can use its dimension-dependent moment bound for each
tensor and sum, or use their maximum as a coarse whole-update omega. This can
be loose for a large layer. Measured errors describe the realised batch but
are not substitutes for a conditional expectation theorem. Floating-point
quantization error would need an additional term for a formal implementation proof.

## 3. One-round descent inequality and where a convergence proof remains open

Let F be a differentiable objective with L-Lipschitz gradient, g=grad F(w),
`U=sum_i a_i u_i`, `q_bar=sum_i a_i(z_i-u_i)`, and `e_bar=sum_i a_i e_i`.
The ideal update is `w_next=w+U+q_bar+e_bar`. Condition on w and all local updates,
and take expectation over ideal codec randomness, so `E[q_bar]=0`.
For any positive rho and eta, smoothness, Young's inequality, and (1) give

```
E[F(w_next)] <= F(w) + <g,U> + (rho/2)||g||²
  + (L/2)(1+eta) (||U||² + E||q_bar||²)
  + [1/(2*rho) + (L/2)(1+1/eta)]
      * sum_i a_i tau_i ||u_i||².                       (4)
```

To verify (4), apply smoothness to the update; bound `<g,E e_bar>` by
`rho||g||²/2 + E||e_bar||²/(2*rho)`; split the quadratic update term into
`U+q_bar` and `e_bar`; use `E||U+q_bar||²=||U||²+E||q_bar||²`.
Then substitute (1). This argument does not cancel `<q_bar,e_bar>` or assume
LLZ errors average to zero. A coarse QSGD term follows by Jensen:
`E||q_bar||² <= sum_i a_i omega_i ||u_i||²`.

Turning (4) into a stationarity or objective-gap rate still requires:

- a bound connecting the multiple-local-step drift term `<g,U>` to global
  descent, with local learning rate, batch noise and empirical client differences;
- a sampling argument matching the implementation's shuffled local epochs;
  unbiased independent minibatch noise must not simply be assumed for random
  reshuffling after conditioning on the already-used batches;
- second-moment bounds on the actual local deltas and QSGD errors;
- choices of rho/eta, learning rates and budgets that permit summing the
  inequality, plus a lower bound on the objective;
- explicit treatment of participation and feedback if either is later changed.

The present frozen policies keep tau constant across clients and rounds, but
their selected p may change. A constant positive tau alone neither proves
convergence nor proves failure. Whether a schedule is necessary depends on the
remaining bounds and assumptions. Neural-network nonconvexity suggests a
stationarity target; identification accuracy remains an empirical metric.

## Questions to resolve with the supervisor

1. Is a nonconvex stationarity bound for local-update FedAvg the intended target,
   or should theory first analyse a simpler one-local-step model?
2. Is the additional-error constraint an acceptable theoretical object, with
   total compression error reported separately?
3. Should the next theoretical extension use a decreasing distortion budget,
   or first attempt a sufficient constant-budget condition?
4. Which heterogeneity/noise assumptions are reasonable for the intended thesis
   claim, rather than inferred from this finite IID-partition demo?

The 22 October [conditional stationarity extension](adaptive-llz-conditional-stationarity.md)
now telescopes (4) under explicit local-alignment and moment assumptions,
provides a sufficient decreasing-budget schedule, and proves a narrower
one-full-gradient-step result. Those local assumptions have not been derived
for the implemented shuffled-local-epoch pipeline. A convergence theorem for
that actual algorithm remains open.
