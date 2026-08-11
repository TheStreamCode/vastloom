# M8 governed CUDA RMSNorm validation

Validation date: 2026-08-11

Implementation commit: `a94aa04df5389dfa4758ce858a14c2be74f807d9`

## Scope and result

This milestone executes Vastloom's first real CUDA math operation while retaining the existing
toolkit-free and Governor-first boundary:

```text
bounded FP32 input + weight
  -> reserve caller input/weight/reference storage
  -> reserve host output storage
  -> reserve one packed device allocation
  -> synchronous host-to-device copies
  -> JIT-load embedded PTX
  -> scalar RMSNorm kernel + context synchronization
  -> synchronous device-to-host copy
  -> finite check + CPU-reference comparison
  -> device free + reservation release
```

The native Windows and WSL binaries both executed two 4,096-wide rows on the reference RTX 4070
Super. Each run copied 49,152 bytes to the device, copied 32,768 bytes back, used one 81,920-byte
packed device allocation and preserved at least 1,073,741,824 bytes of live VRAM headroom. Both
reported `0.000025` maximum absolute error against the existing double-accumulating DeepSeek CPU
RMSNorm oracle, below the declared `0.000030` gate, and released the VRAM allocation.

This is real GPU execution and real host/device transfer. It is one deliberately scalar FP32
reference operator, not a DeepSeek layer, token, generation result or optimized CUDA backend.

## Implemented

- The dynamic driver now resolves synchronous H2D/D2H copy, PTX module, kernel launch and context
  synchronization entry points from `nvcuda.dll` or `libcuda.so.1`; CUDA headers, runtime,
  toolkit and `nvcc` remain unnecessary.
- Driver calls accept only live allocations created by the same backend. Copy and kernel ranges
  must be non-empty, overflow-safe, inside the allocation, FP32-aligned and non-overlapping.
- PTX JIT failures include a bounded NVIDIA error log. The log cannot make the host read beyond its
  fixed 8 KiB buffer even if the driver does not append a terminator.
- `cuda_rms_norm` caps the reference operation at 4,194,304 elements, rejects invalid shapes,
  non-positive epsilon and non-finite input/weights before device work, and checks every output for
  finiteness before publication.
- The host output reservation is acquired before allocation and remains owned by the returned
  vector. One packed Governor-backed VRAM allocation contains input, weights and output.
- H2D, launch and D2H errors leave publication empty and release scratch/VRAM transactionally.
  An external release failure remains fail-closed: the uncertain VRAM charge is retained.
- `cuda-info --rmsnorm-smoke` applies the normal hardware, runtime and process policy, reserves its
  caller-side vectors, generates deterministic FP32 values and compares all 8,192 output elements
  with the CPU oracle.

The PTX assigns one thread to each row. That is intentionally simple and slow; a later optimized
reduction must preserve this result within an explicitly measured tolerance.

## Deterministic fault coverage

The hardware-independent backend suite covers:

- exact copy round-trip and rejection of empty, out-of-range and cross-driver buffers;
- invalid shape, epsilon, alignment, overlap and output range before backend launch;
- exact fake-device RMSNorm equality with the CPU oracle;
- retained host-output accounting until result destruction and zero VRAM after success;
- injected error and C++ exception at H2D, kernel and D2H stages with zero residual accounting;
- invalid/non-finite/oversized input and insufficient scratch refusal before device allocation;
- explicit device-release failure retried during destruction, then charged fail-closed; and
- structured CLI rejection for duplicate, incompatible, missing and oversized smoke options.

The CLI integration test runs the real RMSNorm automatically when a usable CUDA driver is present.
It accepts a clean structured unavailable-driver result on CPU-only CI and skips the live
closed-source driver under sanitizers; the fake backend remains fully instrumented.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,878,086,144 CUDA-reported VRAM bytes;
- NVIDIA driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- CUDA Driver API encoded version 13,030 (`13.3`);
- `nvcuda.dll` and WSL `libcuda.so.1` available;
- no CUDA toolkit or `nvcc` available.

## Exact validation results

All compiler configurations treated project warnings as errors.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 17/17 passed in 14.76 s |
| GCC Debug + real metadata opt-ins | 17/17 passed in 31.36 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 17/17 passed in 96.94 s |
| Clang 18 Release | 13/13 passed in 5.44 s |
| GCC Debug + libstdc++ debug iterators/assertions | 13/13 passed in 9.27 s |
| MSVC Release x64 | 13/13 passed in 5.34 s |
| MSVC Debug x64 | 13/13 passed in 8.80 s |
| TSan focused native binaries | core, range-artifact and CUDA-driver tests passed |
| Python `compileall`, manifest and importer suites | passed; 1 Windows symlink case skipped |
| Windows real RMSNorm smoke | passed, max absolute error 0.000025 |
| WSL real RMSNorm smoke | passed, max absolute error 0.000025 |

TSan binaries used `TSAN_OPTIONS=halt_on_error=1` with `setarch x86_64 -R` to avoid the known WSL
address-layout conflict. ASan/UBSan skipped the live closed-source NVIDIA call while retaining the
fully instrumented fake-driver tests; the same WSL driver incompatibility is detailed in the
earlier CUDA control-plane report. Non-sanitized Windows and WSL binaries executed the real kernel.

## Timing and traffic statement

Five separate native Windows CLI processes took 225.024, 191.098, 172.376, 174.396 and 172.332
milliseconds end to end: 174.396 ms median. Five PowerShell-launched WSL processes took 3,667.480,
362.563, 337.953, 347.725 and 351.858 ms: 351.858 ms median, with the first sample including cold
WSL startup.

These times include process launch, hardware probing, safety-policy construction, CUDA driver and
primary-context work, PTX JIT, two copies, the scalar kernel and CPU reference generation. They are
not kernel latency, bandwidth, tokens/second or a speedup claim. The exact logical transfer count
is 81,920 bytes per smoke. No model weights or physical NVMe traffic were involved.

## Known limitations and next bottleneck

- The kernel is scalar per row and recompiles its PTX module for each operation.
- Copies are synchronous and pageable; there are no streams, events, pinned buffers, cancellation
  points inside driver calls or copy/compute overlap.
- Only FP32 RMSNorm exists. Dense matmul, RoPE, attention, KV state, router, SwiGLU and packed
  IQ1_S/IQ3_XXS expert CUDA kernels remain absent.
- The operation is not connected to a DeepSeek graph or checkpoint payload.
- Long-lived device-pressure monitoring, device loss/reset recovery and multi-GPU placement remain
  unimplemented.
- macOS/Metal, ROCm and Vulkan are separate roadmap tracks.

The next M8 correctness slice is a bounded FP32 dense matvec/GEMM using a committed DeepSeek
fixture, followed by attention/state operators. Packed expert kernels and asynchronous
SSD-to-RAM-to-VRAM overlap come only after those CPU-parity gates pass.
