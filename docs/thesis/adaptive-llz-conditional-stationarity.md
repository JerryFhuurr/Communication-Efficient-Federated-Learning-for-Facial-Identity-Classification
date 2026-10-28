# Adaptive LLZ: conditional stationarity results

Prepared 22 October 2026. These are self-contained derivations for an ideal
real-arithmetic model, extending the [theory outline](adaptive-llz-theory-outline.md).
They are sufficient conditions, not an assertion that the current ResNet18,
shuffled-local-epoch implementation satisfies them. No new training was run.

## 1. Assumptions and conditioning

Let F be differentiable, L-smooth for L>0, and bounded below by F_inf.
Let H_t contain the history up to w_t, before the current local updates and
codec randomness. Client weights a_i are nonnegative, sum to one, and are
known given H_t. Set g_t=grad F(w_t), U_t=sum_i a_i u_i,t and
S_t=sum_i a_i ||u_i,t||². Assume all displayed expectations are finite.

Given H_t and all current local updates, ideal QSGD satisfies

```
E[q_i,t | H_t, updates] = 0,
E[||q_i,t||² | H_t, updates] <= omega ||u_i,t||²,
```

where q_i,t=z_i,t-u_i,t and omega>=0 is a uniform bound. Independence across
clients is unnecessary for the Jensen bounds below. Fresh conditional QSGD
randomness is a mathematical assumption. A fixed PRNG seed is the reproducibility
mechanism in an experiment, not a proof of the stochastic assumption.

The selected LLZ error e_i,t can depend on that same QSGD draw, but must obey

```
||e_i,t||² <= tau_t ||u_i,t||²
```

for every outcome, where tau_t>=0 is known given H_t. For client-specific
budgets, their maximum is a sufficient common tau_t. The byte-minimizing rule
can select any feasible candidate. Its objective need not appear in the proof.
The server performs w_(t+1)=w_t+U_t+q_bar_t+e_bar_t with no error feedback.

The additional assumptions about the actual local updates are

```
E[<g_t,U_t> | H_t] <= -gamma_t c ||g_t||² + gamma_t d_t,      (A)
E[S_t | H_t] <= gamma_t² (A ||g_t||² + B),                   (B)
```

with c>0, A>0, B>=0, d_t>=0 and gamma_t>0. Here the letter A in (B) is a
constant, not an aggregation matrix. Constants and gamma_t are deterministic
for the statements below. The history-dependent nonnegative d_t is integrable.
Assumption (A) expresses alignment of the aggregate delta with global descent.
Assumption (B) controls local delta magnitude. These are substantive hypotheses,
not facts established by the IID label of the demo or by the saved accuracy.

## 2. Conditional descent and a finite-horizon bound

Choose r>0 and eta>0 and define

```
C_t = (1+eta)(1+omega) + (1+1/eta) tau_t,
D_t = c-r/2-tau_t A/(2r)-L gamma_t C_t A/2.
```

Then

```
E[F(w_(t+1)) | H_t]
 <= F(w_t) - gamma_t D_t ||g_t||²
    + gamma_t d_t + gamma_t tau_t B/(2r)
    + L gamma_t² C_t B/2.                                (5)
```

**Proof.** The earlier one-round inequality conditions on H_t and the updates.
Apply it with rho=gamma_t r. Jensen gives ||U_t||²<=S_t and
E||q_bar_t||²<=omega S_t. The pointwise LLZ bound gives
E||e_bar_t||²<=tau_t S_t. Apply the tower property and (A)–(B).
The coefficient of gamma_t ||g_t||² is
-c+r/2+tau_t A/(2r)+L gamma_t C_t A/2. The remaining terms give (5).
This argument retains the potentially nonzero LLZ bias and does not cancel
the QSGD/LLZ cross term. This proves (5).

For constants gamma,tau,d and D>0, summing (5) and using F>=F_inf gives

```
(1/T) sum_(t=0)^(T-1) E||grad F(w_t)||²
 <= [F(w_0)-F_inf]/(gamma D T)
    + [d+tau B/(2r)+L gamma C B/2]/D.                     (6)
```

For tau>0, r=sqrt(tau A) minimizes r/2+tau A/(2r). Thus
sqrt(tau A)<c and a sufficiently small gamma are sufficient for D>0.
This yields a bound to a neighborhood in general. Reducing gamma alone does
not remove the displayed tau B/(2r) term. The bound may be loose and is not
a lower bound on the actual error or evidence that learning fails.
For tau=0 one can remove the LLZ term directly, without a positive-r Young
inequality, and obtain the usual unbiased-compression descent form.

## 3. A sufficient schedule for vanishing expected stationarity

Fix r=c/2 and eta>0. Suppose tau_t is deterministic, nonnegative and bounded,
and for every t

```
tau_t A <= c²/8,
L gamma_t C_t A/2 <= c/8.
```

Then D_t>=c/2. If

```
sum_t gamma_t = infinity,
sum_t gamma_t² < infinity,
sum_t gamma_t tau_t < infinity,
sum_t gamma_t E[d_t] < infinity,
```

then

```
lim_(T->infinity)
 [sum_(t<T) gamma_t E||grad F(w_t)||²] / [sum_(t<T) gamma_t] = 0.  (7)
```

**Proof.** Sum the unconditional version of (5), bound D_t below by c/2,
and bound C_t uniformly using bounded tau_t. The lower bound on F bounds the
sum of gamma_t E||g_t||² by a finite constant, since the other series are
summable by hypothesis. Dividing by the divergent sum of gamma_t proves (7).
Equivalently, independently choose an output index with probability
gamma_t/sum_(s<T)gamma_s. Its expected squared gradient norm tends to zero.
This is a weighted expected-stationarity statement, not convergence of every
iterate, global optimality, or a classification-accuracy guarantee.

One illustrative sufficient choice is gamma_t=g/(t+1)^alpha with
1/2<alpha<=1 and tau_t=k gamma_t, with g small enough for the two inequalities,
provided the alignment residuals satisfy the stated summability condition
(for example d_t=0 or a deterministic O(gamma_t)). This is an example allowed
by the mathematical assumptions, not an implemented or empirically selected
schedule. Fixed positive tau does not satisfy this particular sufficient
schedule. That does not establish that every constant-budget algorithm fails.

## 4. A fully specified special case

For a simpler mathematical reference, compress one full-gradient delta
u_t=-gamma grad F(w_t), using one ideal unbiased QSGD stage and a feasible LLZ
stage. This corresponds to c=A=1 and B=d_t=0 above. It also describes clients
with identical deterministic full-gradient deltas. It does not describe the
demo's multiple minibatch steps on distinct empirical client datasets.

For 0<=tau<1 set

```
C = (sqrt(1+omega)+sqrt(tau))²,
D = 1-sqrt(tau)-L gamma C/2.
```

For 0<gamma<2(1-sqrt(tau))/(L C), D>0 and

```
(1/T) sum_(t<T) E||grad F(w_t)||²
 <= [F(w_0)-F_inf]/(gamma D T).                          (8)
```

**Direct proof, including tau=0.** Conditional on w_t,
E<u_t+q_t+e_t,g_t> <= -gamma(1-sqrt(tau))||g_t||².
This follows from E q_t=0, Cauchy–Schwarz and
E||e_t||²<=tau gamma²||g_t||². The random-vector L2 triangle inequality,
and E||u_t+q_t||²<=gamma²(1+omega)||g_t||², give
E||u_t+q_t+e_t||²<=gamma² C||g_t||². Apply smoothness, telescope and use
F>=F_inf. For several identical deltas the Jensen bounds give the same result
without a claim of a client-count variance improvement. This proves (8).

## 5. Meaning for the implemented experiment

The pilot budgets are 5.9 and 7.0. They are ratios of squared norms, not
percentages. They do not satisfy tau<1 in special case (8). The actual local
training has shuffled minibatches, a full local epoch and client-specific
empirical objectives. We have not established constants c,A,B,d_t or a global
smoothness constant for it. We therefore cannot apply (6)–(8) numerically to
the pilot, claim its convergence, or select a safe learning rate from them.

The useful theoretical progress is explicit: selection preserves a pointwise
error bound, that bound controls aggregate bias, and sufficient alignment,
moment and budget conditions turn it into stationarity bounds. Proving such
alignment/moment conditions from local-epoch dynamics remains the main bridge
to a theorem for the implemented algorithm. Random reshuffling and finite
precision require their own treatment. No claim of novelty or journal-ready
proof certification follows from these derivations.

## Literature relationship and next supervisor decisions

[QSGD](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf)
supplies the ideal first-stage moment assumptions. [FedPAQ](https://proceedings.mlr.press/v108/reisizadeh20a.html)
motivates local-update analysis, but its theorem does not establish our (A)–(B)
or remove LLZ bias. [AdaQuantFL](https://arxiv.org/abs/2102.04487) and
[DAdaQuant](https://proceedings.mlr.press/v162/honig22a.html) establish precedents
for adaptive quantization. Equations (5)–(8) are the derivations above, not
results attributed to those papers. The reviewed assumptions are mapped in
the [outline](adaptive-llz-theory-outline.md).

Ask whether to prioritize deriving local-drift conditions for (A)–(B), or first
use the one-step model as the thesis's initial theoretical result. Agree the
intended stationarity notion and acceptable assumptions before claiming a
theorem for the actual neural FedAvg implementation. No new experiment batch
is authorized by this document.
