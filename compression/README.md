# Standalone QSGD and LLZ-p codecs

`adaptive_llz.select(updates, levels=..., rng=..., budget=..., window_size=128)`
now tries p=0/1/2 on one shared QSGD draw per tensor and selects one p for the
whole update. Its constraint is additional decoded LLZ squared error divided
by original-update squared norm. It counts packet bytes by default; a caller
can supply a complete-message cost callback without introducing a Flower
dependency. See the [precise contract](../docs/thesis/adaptive-llz-contract.md).

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


## Global Top-k with client error feedback

`compression.topk` is a NumPy-only float32 vector codec, separate from QSGD's
symbol registry. It retains exactly k coordinates by descending magnitude;
ascending flattened index resolves ties. Packet indices are stored in ascending
order. `TPK1` framing is little endian: magic (4 bytes), vector length (uint32),
k (uint32), k uint32 indices, then k float32 values: exactly `12 + 8*k` bytes.
Decoding requires the expected length/k and rejects malformed, duplicate,
out-of-range, nonfinite, truncated, or trailing data. Zero vectors still retain
k entries; this baseline has no variable-length zero shortcut.

`federated_compression.topk` sorts tensor names, fingerprints names/shapes/dtypes,
and compresses the concatenated trainable delta globally. Fixed BatchNorm buffers
remain outside compression. Per-client float32 residuals live in Flower
`Context.state`: corrected = delta + residual; next residual = corrected minus
the decoded packet. State binds to run, partition, layout, k, and preceding round.
Missing/stale state fails closed. No model-only resume or same-round retry is
supported: restart an interrupted experiment as a new run from round one.
Residuals are local algorithm state, not application upload payloads; simulation
runtime transport of Context state is outside our logical-message ledger.

The new method is `--compression topk-ef --topk-k 124000`. It uses feedback while
the existing QSGD/LLZ pilot does not. It is a method-level comparison, not an
isolated sparsifier ablation or an STC reproduction. The implementation adapts
[Sparsified SGD with Memory](https://arxiv.org/abs/1809.07599) and
[error feedback](https://proceedings.mlr.press/v97/karimireddy19a.html) to FedAvg
model deltas; their convergence claims are not automatically inherited.

Run the agreed single seed-42, 15-round comparison against the saved study:

```powershell
.venv\Scripts\python.exe -m flower_face.check_topk --baseline outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z
```

The runner saves a source snapshot, configuration, provenance checks, terminal
log, model checkpoints, comparison table, accuracy plot, and timings under
`outputs/topk-ef-<timestamp>/`. It never launches historical baselines again.
An incompatible baseline stops the comparison instead of silently pooling it.
Do not use this command just to inspect existing results: it starts training.

Client encode timing includes delta creation, validation, residual addition,
selection, packing, local decoding, and state persistence. Diagnostic norms are
separately timed. Server decode includes packet checks and tensor conversion,
not aggregation. Round timing spans training configuration through validation
and checkpoint selection, including simulation scheduling. All rounds are kept;
first-round timings are listed separately. These CPU wall times exclude any
claim of a measured physical-network speedup. Historical baselines have no
matched codec timing. Per-client timing metrics add counted message bytes.

To regenerate an existing Top-k report without launching training:

```powershell
.venv\Scripts\python.exe -m flower_face.check_topk --report-only outputs/topk-ef-20260930T113916818351Z
```

Report-only mode can present OS differences explicitly as historical context;
it still rejects changes to initialization, data, library versions, training
settings, or fixed-buffer policy. It never claims an environment match when
one is absent. Scientific interpretation is kept separately in discussion.md and preserved on
report regeneration. A new training run checks the OS before launching; an
explicit --allow-platform-difference is required to use older OS references as
historical context. It does not make the environments equivalent.


## Learned-codec calibration data

The reproducible collector uses the existing DigiFace layer3 feature cache and
fine-tuning initialization. It gathers four clients x two reset one-epoch local
trajectories on training images only. Fitting and reconstruction-validation
trajectories share the image population but have independent recorded seeds;
this is trajectory separation, not an unseen-image or unseen-client evaluation.

```powershell
.venv\Scripts\python.exe -m flower_face.collect_ae_updates
```

This command performs calibration training. It reuses a compatible, hash-verified
completed collection when available. Otherwise it saves a source snapshot and
at most ten cumulative deltas per trajectory under `outputs/ae-calibration-*`.
Fitting stores up to 32,768 sampled normalized 256-value blocks per client;
validation stores up to 8,192 sampled blocks plus full held-out deltas. The data
size is about 1.52 GB for this model. Each tensor is blocked/padded independently.
Saved scales, valid lengths, sorted tensor IDs, block indices, snapshot steps,
and population weights preserve the sampling convention and lineage.

Normalization uses float64 norm accumulation and finite float32 scales. Zero
blocks remain exactly zero. Stratified sampling covers every tensor/snapshot,
then allocates approximately in proportion to block population. Weighted losses
can recover the uniform full-block population; coordinate-weighted or masked
losses must separately account for partial padding. Do not treat padded entries
as additional original coordinates.

The optional `on_batch` observer in `finetune_task.train` captures state after
SGD steps; affected tests verify that read-only observation preserves training.
Collection records preparation, training, snapshot processing, file writing,
and concurrent wall time separately. Calibration is additional work, excluded
from the previous FL runs. There is no Flower traffic measurement in this job.
The collector does not train an autoencoder or evaluate identity validation/test
labels. The full deltas are simulation artifacts for offline codec checks, not a
claim that a deployed server can access private client updates freely.

## Standalone learned codec (v1)

`autoencoder.py` implements a CPU float32 256 -> 64 -> 6 -> 64 -> 256
autoencoder, with ReLU hidden layers and linear latent/output layers. It is
independent of Flower. Training is separate from the packet API:

```python
from compression.autoencoder import BlockAutoencoder, encode, decode, weight_bytes, load_weights

model = BlockAutoencoder().eval()  # fit first for useful reconstruction
packet = encode(named_float32_deltas, model)
decoder_artifact = weight_bytes(model, decoder_only=True)
server_decoder = load_weights(decoder_artifact, decoder_only=True)
restored = decode(packet, server_decoder, expected_shapes)
```

Each tensor is processed independently in sorted name order, normalized using
the calibration convention, padded to 256 and cropped on decode. Zero-norm
blocks have zero latents and decode to exact zero even with decoder biases.
Inference uses chunks of at most 1,024 blocks and requires eval-mode finite CPU
float32 weights. Gradients and model state are unchanged.

`AEP1` packets contain a 44-byte header (magic, layout byte length, block count,
32-byte decoder SHA256), canonical ASCII JSON names/shapes, then 28 bytes per
block: one float32 L2 scale and six float32 latents, little endian. Receivers
must provide trusted expected shapes; packet metadata cannot choose allocation
sizes. Wrong versions, layouts, lengths, decoder identities, negative/nonfinite
scales, nonfinite latents and noncanonical zero blocks are rejected. There is
no latent quantization or entropy coding. The codec is lossy even though its
latent serialization is float32.

`AEW1` weight artifacts use a five-byte version/role header, fixed architecture
weight order, float32 weights and a 32-byte SHA256 checksum, without pickle.
Use `weight_bytes(model)` for an encoder+decoder checkpoint and
`load_weights(bytes)` to restore it. Optimizer state is not stored; this is not
a resumable training checkpoint. A decoder-only artifact is **68,389 bytes per
client**; count all four artifacts plus calibration/fitting costs when assessing
setup overhead. SHA256 identifies contents, not the sender's authenticity.

To check one existing calibration update without training:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_autoencoder.py -q
.venv\Scripts\python.exe -m compression.check_autoencoder --collection outputs/ae-calibration-20261005T122126176771Z
```

The smoke command uses untrained weights and saves its report under ignored
`outputs/ae-codec-*`. It demonstrates interface correctness only. Fitting and
held-out reconstruction comparisons remain separate scheduled batches.
This implementation adapts the client-encoder/server-decoder idea from
[Chandar et al. (2021)](https://arxiv.org/abs/2108.05670); it is not a reproduction
of that paper's model or compression-ratio claims.

### Fixed offline fitting and comparison batch

```powershell
.venv\Scripts\python.exe -m compression.fit_autoencoder --collection outputs/ae-calibration-20261005T122126176771Z
.venv\Scripts\python.exe -m compression.report_autoencoder outputs/ae-feasibility-<timestamp>
```

The fitting command runs the agreed four-client batch: ten epochs, Adam 0.001,
batch 1,024, compressor seed 4242 + client ID, no architecture search. It uses
population-weighted squared errors on normalized blocks, masks padding, and
selects the minimum sampled reconstruction-validation loss. Zero norms follow
the codec's exact-zero bypass. Each full held-out update is then probed with AE,
per-tensor QSGD at 127 levels, and global Top-k at k=124,000 without feedback.
No Flower training or identity accuracy evaluation is launched.

The result directory records full/decoder weight artifacts, per-epoch history,
per-update/per-tensor errors, actual bundle bytes, seeds, hashes and timings.
Pass `--output outputs/ae-feasibility-<timestamp>` to reuse a compatible result
directory; completed client fits/probes can be reused after an interruption.
Source/configuration/environment or artifact mismatches fail closed. The report
command checks saved artifacts and creates a report/plot without training.
Reconstruction-validation updates also select checkpoints; they are a development
set, not an untouched test set. They share training images with fitting updates.

Keep the numerical report separate from the final `decision.md` and weekly
summary in the result directory. A byte saving alone does not justify Flower
integration: consider reconstruction error/direction, setup, CPU cost and
later-round update drift. The feasibility batch ends at the proceed/stop review;
architecture changes and FL confirmation require a separately agreed batch.

### Matched QSGD/LLZ screening batch

```powershell
.venv\Scripts\python.exe -m flower_face.check_compression
# Resume a specific interrupted batch without rerunning completed methods:
.venv\Scripts\python.exe -m flower_face.check_compression --resume outputs/llz-matched-<timestamp>
```

This runner executes only QSGD and QSGD+LLZ at p=0 and p=1: seed 42,
15 rounds, four clients, levels 127, window 128, existing fine-tuning data/model
settings and no feedback. It freezes current source, requires exact p=0 learning
replay, and reports p=1 accuracy/loss versus logical communication cost.
Full-launch wall times include Flower startup/simulation/training/validation;
they are not isolated codec measurements. Outputs, logs and plots are saved
under ignored `outputs/llz-matched-*`. No additional baseline or seed is run.

To extend the completed batch with exactly the two remaining controls:

```powershell
.venv\Scripts\python.exe -m flower_face.complete_comparison --baseline outputs/llz-matched-20261010T114122936542Z
# Reuse completed controls after an interruption:
.venv\Scripts\python.exe -m flower_face.complete_comparison --resume outputs/compression-controls-<timestamp>
```

This reuses the three completed QSGD/LLZ runs and their frozen app, checks current
environment compatibility before training, then runs uncompressed and Top-k+EF
controls with the original settings. The five-method report marks feedback
differences and timing limitations across execution dates. No optimizer, seed,
dataset or budget sweep is added. Results stay under `outputs/`.
