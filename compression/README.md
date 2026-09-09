# Standalone QSGD and LLZ-p codecs

This package requires NumPy and Python's standard library. It imports neither
Flower nor PyTorch. It implements `qsgd-l2-dense-v1`, `llz-p-fixed-v1`, and the
composed `qsgd-llz-p-v1` packet. The separate
`federated_compression/updates.py` adapter uses these codecs for optional client
model-delta uploads. The commands below remain standalone synthetic codec checks.

## Quantization and encoding are separate

The quantizer follows Section 3.1, Equation (4), of
[Alistarh et al., QSGD, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf).
For a nonzero input tensor flattened to a vector, compute its L2 norm `r`. For
each coordinate, let `a = s * abs(x_i) / r`, `l = floor(a)`, and round the magnitude
to `l + 1` with probability `a - l`, otherwise to `l`. Retain the sign and
reconstruct with `r * signed_level / s`. Zero inputs reconstruct as zero.

In exact arithmetic this quantizer preserves each coordinate in expectation.
Floating-point norms, arithmetic, and the output dtype introduce rounding;
finite Monte Carlo checks are evidence of correct implementation, not a proof.
The theoretical squared-error bound is
`min(n/s**2, sqrt(n)/s) * ||x||_2**2` for a vector of `n` coordinates.

**The project's byte encoding is dense fixed-width, not the paper's Elias
encoding.** We make no claim to reproduce that encoding's communication bounds
or the paper's distributed-SGD convergence results in our FedAvg demo.

`s` counts positive quantization intervals, not bits. Magnitudes range from 0 to
`s`, inclusive. Each coordinate occupies `1 + s.bit_length()` bits: one sign bit
and enough magnitude bits for the endpoint `s`. Thus `s=127` uses eight bits per
coordinate, while `s=128` uses nine. Smaller `s` usually trades a smaller packet
for greater quantization error; higher accuracy in training must be measured.

## API

```python
import numpy as np
from compression.qsgd import encode, decode, packet_stats

rng = np.random.default_rng(42)
x = np.array([3.0, -4.0, 0.0], dtype=np.float32)
payload = encode(x, levels=127, rng=rng)
reconstructed = decode(payload)
print(packet_stats(payload))
```

- `quantize(array, levels=..., rng=...)` returns a `QuantizedTensor`: signed integer
  `codes`, an L2 `norm`, `levels`, and the original floating `dtype`. Shape belongs
  to `codes`. One norm is used per call, not per coordinate.
- `dequantize(q)` reconstructs floating values from those integers.
- `pack(q)` and `unpack(payload)` preserve the quantized integers and metadata
  exactly. These functions add no further quantization loss.
- `encode` combines quantization and packing; `decode` combines unpacking and
  reconstruction. Decoding requires only the packet, with no RNG or original tensor.
- `packet_stats(payload)` validates the packet and counts all its bytes and bits.

Use an explicit NumPy Generator. Reusing it advances the random stream; creating
the same seed anew reproduces the same packet for the same input and NumPy
implementation. Future clients will need independent streams per client/round,
separate from model and data-loader randomness.

Inputs are finite float32 or float64 arrays, including scalars, empty arrays,
and noncontiguous views. Internal computation and the stored norm use float64.
Reconstruction restores shape and floating precision in native byte order. Signed zero is
canonicalized to positive zero. Invalid levels, unsupported types, nonfinite
values/norms, and unrepresentable output values are rejected. The codec has no
error feedback, optimizer state, residual accumulation, or model-specific logic.

## QSD1 packet format

All header fields are little-endian, with no native alignment padding.

| Field | Bytes | Definition |
| --- | ---: | --- |
| Magic/version | 4 | ASCII `QSD1` |
| Output dtype | 1 | 1 = float32, 2 = float64 |
| Dimension count | 1 | 0 through 32; 0 denotes a scalar |
| Levels `s` | 2 | Unsigned integer, 1 through 65535 |
| L2 norm | 8 | IEEE float64, finite and nonnegative |
| Shape | `4 * ndim` | Unsigned uint32 for each dimension |
| Codes | `ceil(n * (1 + s.bit_length()) / 8)` | Dense packed sign/magnitude symbols |

Tensor coordinates use C order. Each symbol is sign (1 = negative), then magnitude
bits, most significant first. Symbols are concatenated and packed most significant
bit first into bytes. Any unused trailing bits are zero. Negative-zero codes and
unused magnitude values above `s` are rejected. Scalars have one coordinate;
empty shapes containing a zero dimension have no codes and a zero norm.

Total bytes are exactly
`16 + 4*ndim + ceil(n*(1+s.bit_length())/8)`, even for all-zero tensors. There is
no hidden sparse encoding, entropy coder, checksum, or zero-vector shortcut.
Truncated/trailing data and invalid metadata or padding are rejected, but a
valid-looking bit flip is not guaranteed to be detected; this is not a checksum.

`packet_stats` separates metadata bytes, code bits, and final padding bits.
Its `encoded_bits` is always `8 * len(payload)`, including padding. Tiny tensors
can expand because their headers dominate. This packet contains one unnamed
tensor; a future model/update container must additionally count tensor names,
ordering, framing, and Flower message metadata. Packet sizes here must not be
compared directly with the earlier complete Flower object-graph totals.

`qsgd_llz.pack` passes the already sampled QSGD integer codes to LLZ-p, so it
does not quantize again or consume additional randomness. At `p=0`, LLZ decoding
restores every QSGD code exactly. The combined decoder therefore reconstructs
the same floating update as the dense QSGD packet. LLZ framing, metadata, and
byte padding are included in `packet_stats`; see `papers/README.md` for the
paper mapping and source fingerprints.

## Checks

Verified on 2026-09-08: 38 QSGD test cases passed; the complete project suite passed
all 51 cases. A fresh-process import confirmed that the codec loads neither
Flower nor PyTorch. The default synthetic report is saved in
`outputs/qsgd-codec-20260908T102827179310Z/metrics.json`.

From the project root in PowerShell:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_qsgd.py -q
.venv\Scripts\python.exe -m compression.check_qsgd
.venv\Scripts\python.exe -m pytest tests/test_llz_p.py -q
.venv\Scripts\python.exe -m compression.check_llz --levels 127 --p 0 1 --window-size 128
```

The demo uses a fixed synthetic float32 vector with 4,096 coordinates. It performs
one codec realization per setting, writes a JSON report under
`outputs/qsgd-codec-<timestamp>/`, and performs no training or dataset evaluation.
For a custom check:

```powershell
.venv\Scripts\python.exe -m compression.check_qsgd --size 8192 --levels 7 31 127 --seed 42
```

Verified default check (seed 42):

| `s` | Complete packet bytes | Raw bytes / packet bytes | Relative squared error |
| --- | ---: | ---: | ---: |
| 1 | 1,044 | 15.693 | 55.912244 |
| 7 | 2,068 | 7.923 | 6.712268 |
| 31 | 3,092 | 5.299 | 0.688245 |
| 127 | 4,116 | 3.981 | 0.042619 |

The original vector occupies 16,384 raw bytes. Relative squared error means
`||decoded - original||_2**2 / ||original||_2**2`; it is not a classification error
rate. Results above are a single random draw for each setting. Compression ratios
describe these codec packets only, not end-to-end communication savings.
