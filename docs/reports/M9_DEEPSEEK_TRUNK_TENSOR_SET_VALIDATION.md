# M9 DeepSeek governed trunk tensor-set validation

Validation date: 2026-08-11

Implementation commit: `2a5f1273d4c5fd85d073bea5a29f6888beae994f`

## Scope

This milestone closes the synthetic multi-range artifact-to-graph gate left open by the
single-range loader:

```text
exact non-executing DeepSeek trunk-artifact binding
  -> require one complete range for every non-routed tensor in the layer catalog
  -> reserve aggregate RAM plus maximum single-range staging
  -> read all independently hashed ranges directly into one packed buffer
  -> publish RAM_READY only after every post-read SHA-256 matches
  -> bind typed matrix and I32 views without copying payload
  -> execute the complete reduced mixed-format prefill/decode graph
  -> release one aggregate lease and evict cleanly
```

The executable test is deliberately synthetic and contains 2,644 trunk bytes. No byte from the
official 138,867,160-byte layer-0 trunk or 82.54 GB checkpoint was downloaded or executed.

## Implemented

The model-independent `load_bound_range_set` primitive now:

- accepts a closed ordered set of at most 4,096 non-empty immutable range identities;
- applies caller-lowerable defaults of 512 MiB total and 64 MiB per range, with absolute
  ceilings of 4 GiB total and 1 GiB per range;
- validates every regular non-symlink source, in-file bound, SHA-256 identity, unique logical ID
  and overflow-safe aggregate byte total before changing residency;
- requires one matching `NVME` residency whose object size equals the full packed destination;
- reserves the complete destination in RAM and only the maximum single-range staging allowance
  through the aggregate host-memory gate before allocation or I/O;
- reads each range directly into its final checked subspan, avoiding a second payload-sized copy;
- hashes each destination subspan before publication and invalidates the entire transaction on
  any read, hash, callback, cancellation or allocation failure;
- publishes `RAM_READY` only after all ranges pass, releases staging, and exposes exact packed
  slices through one shared immutable buffer and one `ResidencyUse` lease;
- reports logical bytes, range count, elapsed time and optional process/device counter deltas
  without interpreting them as physical NVMe traffic.

The DeepSeek adapter additionally reconstructs the complete set of non-routed tensors for the
bound layer from the separately verified catalog. It refuses missing, duplicate, split,
non-final, wrong-role, wrong-GGML-type or wrong-source tensors before allocation. Exact shape,
dtype, catalog index, packed offset and byte count remain attached to each retained view.

The reference graph can bind supported two-dimensional F32/BF16/Q8_0/Q5_K/Q6_K/IQ1_S/IQ3_XXS
matrices and I32 tables from that set. It infers physical rows/columns from catalog shape,
requires the caller's unpadded logical width, rejects I32-as-matrix confusion and retains the
aggregate Governor lease without copying encoded payload.

## Correctness and fault coverage

The generic transaction fixture packs four independently hashed artifact ranges totaling 14
bytes into one buffer. The test proves:

- exact slice offsets and lengths;
- one 14-byte RAM reservation plus only the four-byte largest-range staging reservation;
- one retained use regardless of range count;
- rejection of duplicate identities before residency transition;
- a one-byte aggregate-host shortfall fails before allocation or I/O;
- late cancellation and same-size post-binding corruption move the transaction to `FAILED` and
  release every RAM/staging reservation;
- eviction remains blocked while a consumer retains the shared buffer and succeeds afterward.

The architecture preflight uses two artifacts. A split ten-byte tensor across three ranges is
rejected in `NVME`. A second binding with one complete range per catalog tensor loads
successfully, while missing, duplicate-name, wrong-type and non-final mutations all fail before
residency changes.

The reduced DeepSeek integration then republishes all 12 non-routed Q8_0/I32 entries in its packed
layer catalog as separate hashed range files. One transaction retains their 2,644 bytes in a
single buffer. The existing four indivisible IQ1_S/IQ3_XXS expert bundles supply the remaining
5,536 bytes. The resulting graph consumes:

- 11 Q8_0 trunk/router/shared-expert matrices;
- eight IQ1_S routed W1/W3 matrices;
- four IQ3_XXS routed W2 matrices;
- one typed I32 hash table.

All 23 packed matrices and the I32 table match the independently dequantized oracle for prefill
and one incremental decode step, including logits, greedy tokens, expert IDs/weights and
attention indices. The tensor-set wrapper is destroyed before inference; graph descriptors alone
retain one active aggregate lease. Clearing those descriptors permits eviction and returns RAM
and staging reservations to zero.

## Validation host and exact results

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM and NVIDIA GeForce RTX 4070 Super with 12 GiB VRAM;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- GCC 13.3.0, Clang 18.1.3 and Visual Studio 2022/MSVC 19.44;
- no CUDA code and no official DeepSeek payload were executed.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release, warnings as errors, existing real-metadata opt-ins | 15/15 passed in 13.85 s |
| GCC Debug, warnings as errors, existing real-metadata opt-ins | 15/15 passed in 49.70 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 15/15 passed in 78.34 s |
| Clang 18 Release, warnings as errors | 11/11 passed in 5.11 s |
| GCC Debug + libstdc++ debug iterators/assertions | 11/11 passed in 8.62 s |
| MSVC Release x64, warnings as errors | 11/11 passed in 5.42 s |
| MSVC Debug x64, warnings as errors | 11/11 passed in 8.91 s |
| Python manifest suite | 4/4 passed |
| Python importer suite and `compileall` | passed |

ASan used leak detection and halt-on-error; UBSan halted on the first finding. A focused TSan run
passed once, but an immediate full rerun aborted multiple binaries before `main` with
`FATAL: ThreadSanitizer: unexpected memory mapping`. The WSL TSan result is therefore considered
nondeterministic environmental evidence and is not listed as a passing validation gate.

## Performance statement

No tokens/second, physical-device bandwidth, bytes/token or speedup is reported. The executable
artifact fixture totals only 8,180 bytes and is not representative of an official checkpoint or
NVMe pipeline. Its purpose is ownership, integrity, transactionality and numerical parity.

## Known limitations

- The reduced tensor catalog contains the packed layer matrices and I32 route table; remaining
  embeddings, normalization vectors, attention state and output tensors still use the existing
  deterministic fixture storage.
- The graph has resident packed-matrix and I32 bindings, but not yet general retained
  F32/BF16 vector/scalar views needed by every official trunk tensor.
- The local artifact remains a partial non-executing projection and does not verify a complete
  source shard or checkpoint.
- No official trunk range was acquired, decoded or executed.
- Official `tid2eid` routing authority, one official trunk plus selected-expert layer/token,
  generation, CUDA, pinned host memory and physical NVMe overlap remain unvalidated.

## Next bottleneck

Add retained typed F32/BF16 vector/scalar consumers so every tensor in the real 21-range
layer-0 trunk plan can bind to the checkpoint-shaped graph. After that code passes synthetic
full-set faults and the user explicitly authorizes the bounded transfer, import exactly the
138,867,160-byte official layer-0 trunk, reverify every range, execute its authoritative router
for a pinned token and acquire only the six selected complete experts.
