# Source papers and LLZ implementation notes

Both supplied PDFs were read locally on 2026-09-08. Their originals remain in the
user's paper folder; they are not bundled into the application or copied into Git.

## References

1. Huiru Zhong, Kai Liang, and Youlong Wu, *Lossy Lempel-Ziv Coding for Federated
   Learning*, IEEE Globecom Workshops, 2023, pp. 914-919.
   DOI: https://doi.org/10.1109/GCWkshps58843.2023.10464660
   Local file:
   `C:/Users/fhuur/OneDrive/DTU/MasterThesis/Papers need to read/Lossy_Lempel-Ziv_Coding_for_Federated_Learning (1)(1).pdf`.
   SHA256: `6a5a9a48d0ab338824640724243811cdc57aba5668fc70d016b1a26cc70c53d0`.
2. Yue Cao, Kai Liang, Shuoyao Wang, Huiru Zhong, and Youlong Wu,
   *Federated Learning with Lossy Lempel-Ziv Gradient Compression*.
   Supplied revision-1 submission bundle, generated June 2, 2026; treat as a
   manuscript, not as evidence of journal acceptance. Local file:
   `C:/Users/fhuur/OneDrive/DTU/MasterThesis/Papers need to read/Paper(1).pdf`.
   SHA256: `048076c4ac508efce1c51d2673636c01e7c5a3235767b3e168765f8a7688611a`.
   The clean main manuscript occupies PDF pages 4-18 (printed pages 1-15).
   Later pages contain tracked changes, the conference version, and author responses.

Use the revised manuscript's Section III, Equation (5), Algorithms 2/3, and
worked example (PDF pages 7-9) as the LLZ-p implementation reference. The older
conference pseudocode has ambiguous initialization/indexing and decoder handling
of the following literal. The revised description supplies bounded dictionary
search, relative positions, an explicit final literal, and payload field widths.
Its Algorithm 2 still writes `g` in dictionary expressions after initializing a
separate reconstructed sequence; follow the explicit reconstructed-dictionary
prose and worked example rather than interpreting those expressions as immutable
original input. The implementation tests encoder/decoder synchronization directly.

## How the two papers relate

The conference paper introduces LLZ-p as a second compression stage after QSGD.
The revised manuscript adds LLZ-SI, using historical gradient information shared
by the encoder and decoder. Read Section II for the symbol/float distinction,
Section III for LLZ-p, Section IV for LLZ-SI, and Section V for the assumptions
and distortion analysis. Section VI describes experiments and error feedback.

In the revision, `p` bounds differences between signed quantization symbols;
it is not a percentage, a bit count, or an absolute floating-point tolerance.
The QSGD convention also differs: the revision uses `s-1` intervals, whereas
our existing implementation and the original QSGD equation use `levels`
intervals. Therefore **revision s = project levels + 1**. Our `levels=127`
has the symbol alphabet [-127, 127], cardinality 255, corresponding to revision
`s=128`. The existing quantizer is unchanged.

LLZ-p at `p=0` introduces no additional distortion. At positive `p`, it can
introduce bias. The revision's Lemma 1 (PDF page 12) bounds the additional
dequantized squared error by `mismatched_count * (p * norm / levels)^2` for
one tensor before final floating-point rounding. Section V explicitly analyzes
QSGD-LLZ-p without error feedback; LLZ-SI is treated as an empirical extension.
Do not transfer that theorem automatically to our local-epoch FedAvg model-delta
training, which differs from the paper's stochastic-gradient update rule.

## Implemented LLZ-p semantics

`compression/llz_p.py` implements independent integer-symbol encoding/decoding.
`compression/qsgd_llz.py` preserves QSGD reconstruction metadata around that
packet. Neither module imports Flower or PyTorch.

- Start with an empty dictionary; retain the last `window_size` reconstructed
  symbols. Reset the dictionary for each packet/tensor.
- For every candidate start in the current dictionary, find a longest match
  whose coordinate-wise absolute symbol error is **at most p**.
- Match only already reconstructed symbols; references cannot overlap the
  lookahead. Reserve one input symbol as a literal following every match,
  including the final phrase. This follows the revised bounded search.
- Select maximum length, then the largest relative position on a tie. This
  reproduces Algorithm 2's ascending candidate enumeration with last-match
  overwrite. The implementation scans starts in reverse for efficiency; an
  independent length-first exhaustive reference checks equivalence.
- Emit `(position, length, literal)`, with position one-based within the current
  window. A literal-only token has position and length both zero. Append copied
  reconstructed symbols and the literal to the encoder's reconstructed sequence.
- Decode with the same window, copying exactly `length` symbols and then the
  literal. Input arrays are never modified.

For the manuscript's sequence `[1,1,-1,2,-1,0,2,-1,-1,0]`, `p=1` and window 5,
the encoder emits `(0,0,1), (1,1,-1), (2,2,0), (3,3,0)` and decodes to
`[1,1,-1,1,-1,0,1,-1,0,0]`, matching the supplied example exactly.

## Packet choices made by this project

The paper specifies triplet field widths but not a complete interoperable wire
format. `LLP1` and `QLP1` below are project formats, not author-provided formats.

The core alphabet is [-a, a], where `a=alphabet_bound`. The API accepts signed
integer arrays and returns canonical int32 symbols, preserving values and shape,
not the original integer storage dtype. Integer `p` is restricted to [0, 2a];
for integer symbols a noninteger threshold is equivalent to its floor. Window
sizes are restricted to [1, 65535]. Small/empty tensors are supported even when
the configured window exceeds their length; only available history is searched.

The 23-byte little-endian LLP1 header, struct `<4sBHIIII`, contains:

| Field | Bytes |
| --- | ---: |
| Magic/version `LLP1` | 4 |
| Number of dimensions | 1 |
| Alphabet bound `a` | 2 |
| Distortion threshold `p` | 4 |
| Configured window size | 4 |
| Maximum emitted match length `Lmax` | 4 |
| Triplet count `c` | 4 |

Unsigned uint32 shape dimensions follow. Every triplet then uses the paper's
fixed widths: `ceil(log2(window+1))` position bits,
`ceil(log2(Lmax+1))` length bits, and `ceil(log2(2a+1))` literal bits.
A literal is represented by the unsigned value `literal+a`. Fields are packed
MSB first, without per-token alignment, followed by zero padding to a byte.
If all tokens are literals, `Lmax=0` and the length field has zero bits.
The shape determines the exact decoded length, including scalar and empty cases.

QLP1 adds a 13-byte header, struct `<4sBd`: magic/version `QLP1`, a dtype code
(1=float32, 2=float64), and the original QSGD float64 norm. The nested LLP1
packet supplies shape and `levels=a`. Complete combined metadata therefore
costs `36 + 4*ndim` bytes. No metadata is assumed free or added twice.

Both decoders reject malformed versions, shapes, truncated/extra payload, invalid
references, unused literal codes, inconsistent lengths, and nonzero padding.
The default decoder allocation limit is 10,000,000 symbols and can be overridden
explicitly. This is a format/resource check, not authentication or a checksum.
There is no entropy coding, automatic raw fallback, error feedback, or LLZ-SI in
this implementation. Incompressible inputs can expand; report that honestly.

## Comparison protocol

`python -m compression.check_llz` quantizes each source/level pair once and reuses
the exact codes for QSGD dense packing and QSGD+LLZ-p. It counts every header and
padding bit, checks `p=0` for identical codes and reconstructed float bytes, and
records extra symbol error, Hamming error, and squared reconstruction error.
Inputs, packets, code arrays, hashes, and a JSON report are saved under `outputs/`.
Default inputs are synthetic; `--input` accepts a float32/64 `.npy` array.
These are standalone packet measurements, not Flower messages or network traffic.
