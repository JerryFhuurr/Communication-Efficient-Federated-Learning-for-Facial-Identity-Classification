# LLZ-p paper-to-code audit

Audit date: 25 September 2026. Scope: standalone LLZ-p, QSGD composition,
Flower model-delta integration, and communication accounting. This is a source
and focused-test audit, not an independent reproduction of the paper's learning
experiments or a formal verification of its convergence proof.

**Verdict:** no blocking codec correctness defect was found under the revised
manuscript's reconstructed-dictionary interpretation. No algorithm or packet
changes were needed. The focused batch passed **72 tests**. The known notation
ambiguity and project adaptations below must accompany any fidelity claim.

## Reference and audited version

The implementation reference is Cao et al., *Federated Learning with Lossy
Lempel-Ziv Gradient Compression*, supplied revision-1 bundle generated 2 June
2026. The clean manuscript is PDF pages 4-18. Algorithm details come from Section
III, Equation (5), Algorithms 2-3, and the example on PDF pages 7-9; distortion
comes from Lemma 1 on PDF page 12. The earlier Zhong et al. conference paper
provides context, not a second interchangeable pseudocode specification.

Source PDF hashes and publication-status caveats are in
[papers/README.md](../../papers/README.md). The revision hash is
`048076c4ac508efce1c51d2673636c01e7c5a3235767b3e168765f8a7688611a`.
The PDF passages and rendered algorithm/lemma pages inspected during literature
preparation were reused. The audit checked the actual current working-tree code;
some integration changes are still uncommitted. File hashes and test results are
recorded in `outputs/llz-paper-audit/audit.json`. A Git HEAD alone does not identify
this audited working tree.

## Paper-to-code mapping

| Requirement / source | Implementation | Evidence and verdict |
|---|---|---|
| Signed quantization symbols; revision uses s-1 intervals (Section II) | `qsgd.quantize`, `qsgd_llz.pack` | Project `levels = s-1`; 127 intervals means alphabet [-127,127]. LLZ receives integer codes, not floating weights. Matches the stated convention after translation. |
| Empty initial reconstructed dictionary (Algorithm 2, lines 1-2) | `llz_p._parse` initializes `reconstructed=[]` and pointer 0 | Literal-first behavior and manuscript example covered. |
| Sliding window (Algorithm 2, line 5) | `first=max(0,pointer-window)` | The paper's one-based first index minus one equals `first`. Only the last window symbols are searched. |
| Coordinate error at most p (Equation 5) | `abs(source[pointer+length]-reconstructed[start+length]) <= p` | Integer comparison, Python integers avoid subtraction overflow; exhaustive small cases and cascading-distortion test pass. |
| Longest valid match; last candidate wins equal-length ties (Algorithm 2, lines 6-9) | Descending start search, replace only when length is strictly larger | Equivalent to ascending length/start loops with overwrite. Independent length-first reference agrees. |
| Match entirely in past dictionary; leave a following literal | `limit=min(pointer-start,count-pointer-1)` | Prevents overlap and reserves the final symbol. Boundary, sliding-window, and malformed-reference checks pass. |
| Relative one-based position and literal-only zero fields | `position=best_start-first+1`, or zero for no match | Decoder uses `first+position-1`; literal-only tokens require position zero. |
| Reconstructed-history synchronization | Copy reconstructed match and append original next literal in encoder and decoder | Paper example and independent exhaustive oracle pass. See notation qualification below. |
| Fixed triplet widths (Section III-B, PDF page 8) | `window.bit_length()`, `maximum.bit_length()`, `(2*bound).bit_length()` | Equal to ceil(log2(window+1)), ceil(log2(Lmax+1)), ceil(log2(2*bound+1)). Known packet bytes, zero-width length field, and padding tests pass. |
| Decoder copies match then appends literal (Algorithm 3) | `llz_p.unpack` | Extent and reference checks precede copying; exact output count and observed Lmax are checked. |
| p=0 adds no distortion | `qsgd_llz.pack/unpack`, common `qsgd.dequantize` | Codes, metadata, shape, and reconstructed float bytes agree with QSGD-only packets; no second RNG draw. |
| Additional error bound (Lemma 1) | Shared norm/levels; bounded decoded code differences | Existing test checks squared dequantized error against mismatched-count times squared level step. This is finite numerical evidence, not a convergence proof. |
| Second-stage error measured against QSGD | Adapter `encode_update_with_stats`; server `aggregate_llz_distortion` | Actual packet is decoded for diagnostics; separate denominators for QSGD and LLZ; invalid reported symbol bound rejected. |
| Flower learning replay at p=0 | `decode_update`, `QSGDFedAvg`, `check_llz.compare` | Focused multi-tensor/multi-round checks and fixed-buffer handlers pass; existing 15-round seed-42 replay reused as integration evidence. |

Code: [LLZ core](../../compression/llz_p.py),
[composition](../../compression/qsgd_llz.py), [QSGD](../../compression/qsgd.py),
[adapter](../../federated_compression/updates.py),
[server](../../flower_face/server_app.py),
[replay comparison](../../flower_face/check_llz.py).

## Why reconstructed history matters

Algorithm 2 initializes a separate reconstructed sequence, but still writes `g`
in some dictionary expressions. The prose and worked example explicitly require
reconstructed symbols. The implementation follows that interpretation. Following
immutable original history instead could make the encoder reference values the
decoder does not possess. This is a documented interpretation of inconsistent
notation, not an unqualified claim of literal pseudocode identity. Ask the
supervisor/authors to confirm this interpretation if exact author correspondence
is required; it does not require stopping the present implementation.

An induction explains the implemented guarantee. Initially both dictionaries
are empty. Assuming equal reconstructed histories before a phrase, both sides
copy the same past symbols and append the same transmitted literal. Their new
histories are therefore equal. Each copied source coordinate was checked against
that reconstructed symbol within p; literals have zero error. Hence every
decoded coordinate differs from the original input symbol by at most p. At p=0
the reconstructed sequence is identical. This argument concerns codec symbols,
not the optimizer's convergence or robustness to malicious clients.

The manuscript example gives tokens `(0,0,1), (1,1,-1), (2,2,0), (3,3,0)`
for input `[1,1,-1,2,-1,0,2,-1,-1,0]`, p=1, and window 5. The implementation
returns these tokens and reconstruction `[1,1,-1,1,-1,0,1,-1,0,0]` exactly.

## Distortion and floating-point qualification

For original quantized symbols q and decoded symbols q-hat, let m count unequal
coordinates and let r be the stored tensor norm. Before final dtype rounding,

```text
||dequantize(q-hat) - dequantize(q)||^2 <= m * (p*r/levels)^2.
```

For float32/float64 outputs, each reconstruction is rounded; numerical checks
need an appropriate floating-point tolerance. At p=0 both paths use identical
codes, norm, dtype, and dequantizer, so they are checked by byte equality rather
than an approximate tolerance. QSGD itself remains lossy relative to the local
full-precision update. Positive-p LLZ need not be unbiased.

The decoder cannot prove a p-bound against an unavailable original source.
It validates framing and references; the trusted client computes source-relative
distortion and the server checks the reported maximum. These checks are not
authentication or Byzantine-robust validation.

## Explicit implementation adaptations and limitations

1. **Packet format:** LLP1 and QLP1 framing, little-endian headers, C-order tensor
   flattening, shapes, dtype codes, and byte padding are project choices. The
   manuscript specifies triplet widths but not an interoperable complete format.
   LLP1 adds `23 + 4*ndim` bytes; QLP1 adds another 13 bytes. Complete size is
   `8*(36+4*ndim) + triplet_bits + padding_bits` bits. Tests reconcile this with
   the actual packet length. No hidden entropy coder or raw fallback exists;
   incompressible packets can expand.
2. **Input extensions:** integer p is restricted to [0,2*alphabet_bound]; larger
   values cannot admit additional matches within the alphabet. Scalar, empty,
   noncontiguous inputs and windows exceeding a short tensor's length are
   supported explicitly. Decoded symbols use int32, preserving values/shape,
   not the original integer storage dtype. The default allocation guard is
   10 million symbols; callers can override it. These are API/resource choices.
3. **Memory and complexity:** the search window is bounded, but the encoder
   retains full source/reconstruction lists and all triplets. It is not an
   O(window)-memory streaming encoder. For d symbols and window w, a conservative
   source-level search bound is O(d*w^2), with O(d) working storage plus output;
   early exits and long phrases can reduce actual work. No timing result is
   inferred from this bound. Profiling and codec timing are the next task.
4. **QSGD encoding and tensor boundaries:** the original QSGD quantizer is used
   with a dense fixed-width packet, not its original Elias coding. Each tensor
   has its own norm and fresh LLZ dictionary; there is no cross-tensor or
   cross-round dictionary. Do not attribute original QSGD rate guarantees to
   this packet or assume a flattened whole-model codec would behave identically.
5. **Training protocol:** this implementation compresses local-epoch FedAvg
   model deltas, with fixed BatchNorm buffers. It does not reproduce the
   manuscript's stochastic-gradient update rule. Error feedback and LLZ-SI
   are absent. The manuscript's experimental feedback policy and its theorem
   assumptions must not be silently transferred to these runs.
6. **Measurement:** the Flower ledger counts complete logical message object
   graphs, including configuration/metrics and two full-state download phases
   per round (training and validation). This differs from standalone codec bits
   relative to raw FP32 gradients and is not captured transport traffic. See
   [the experimental protocol](experimental-design.md#5-metrics-and-measurement-boundaries).

## Validation performed

One focused batch, no new training runs:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_llz_p.py tests/test_fixed_buffers.py tests/test_qsgd_flower.py::test_llz_wire_updates_are_exactly_equal_to_qsgd_across_rounds_and_tensors tests/test_qsgd_flower.py::test_llz_secondary_distortion_is_measured_against_same_qsgd_draw tests/test_qsgd_flower.py::test_server_aggregates_llz_distortion_by_coordinates_and_validates_bound -q --junitxml=outputs/llz-paper-audit/tests.xml
```

Result: **72 passed**, with two dependency deprecation warnings from Typer/Click.
The exhaustive test enumerates all ternary sequences of lengths 0-6 for p=0/1
and windows 1/3: 4,372 configurations. Additional tests cover the manuscript
example, maximum alphabet/window, malformed packets, immutable inputs, RNG
consumption, standalone imports, dequantized bounds, integration, and buffers.
The independent parser is an independently structured implementation of the
same interpreted rule, not an author-supplied reference implementation.

The already completed fine-tuned-model experiment verifies exact QSGD replay
at p=0 for 15 rounds and the selected checkpoint. Its numerical results remain
unchanged and are documented in [the protocol](experimental-design.md#4-completed-evidence).
This audit does not certify all possible inputs, platform-independent bitwise
reproducibility, or published training performance.

## Next implementation milestone

Proceed to timing instrumentation with separate encoding, diagnostic decoding,
server decoding, and round-duration measurements. Preserve the current codec
semantics and existing source snapshots. If optimization is subsequently needed,
require identical tokens/bytes and p=0 replay before combining its results with
the current method. Feedback and other algorithmic changes remain separately
named, future experiments rather than fixes inferred from this audit.
