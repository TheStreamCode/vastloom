# M8 native CUDA driver and governed-allocation validation

Validation date: 2026-08-11

Implementation commit: `1ae96fac50f6714dc57af969dbcb4bc09f773eba`

## Scope and result

This milestone closes the first two CUDA control-plane gates without introducing a build-time CUDA
dependency:

```text
system CUDA driver
  -> bounded dynamic symbol binding
  -> context-free device capability probe
  -> Governor VRAM reservation
  -> primary-context retain + live free-memory reserve check
  -> bounded device allocation
  -> move-only ownership
  -> device free + context/Governor release
```

The native Windows and WSL binaries independently identified the reference GPU as an NVIDIA
GeForce RTX 4070 Super with compute capability 8.9, 56 multiprocessors, 1,024 maximum threads per
block, unified addressing and 12,878,086,144 total VRAM bytes. Both completed and released an
explicit 4,194,304-byte allocation while preserving a 1,073,741,824-byte live VRAM reserve.

This is real CUDA driver interaction and real VRAM allocation. It is not a CUDA kernel result,
model execution result, transfer benchmark or claim that DeepSeek runs on the GPU.

## Implemented

- Windows loads `nvcuda.dll` only from `System32`; Linux loads `libcuda.so.1` through the normal
  dynamic-linker path. Missing libraries or required driver entry points fail with structured
  `unsupported` errors.
- The driver boundary uses no CUDA headers, CUDA runtime, toolkit or `nvcc`. The tested machine did
  not have `nvcc` installed.
- `cuda-info` reports the encoded/text driver API version, exact device name, total VRAM, compute
  capability, multiprocessor count, maximum threads per block and unified-addressing support.
- Capability-only inspection does not retain a primary context and does not allocate device memory.
- `CudaDriver::allocate` validates device/size/tag, reserves the requested bytes from the shared
  `ResourceGovernor` before calling the backend, then rechecks live free VRAM with the selected
  primary context current.
- `CudaAllocation` is move-only and owns its driver backend, device pointer, retained primary
  context and Governor reservation. Explicit release is retryable when the backend has not freed
  the pointer.
- Backend allocation failures and malformed handles roll back transactionally. If an external
  allocation cannot be cleaned up, its Governor bytes deliberately remain charged and the driver
  module is retained until process teardown rather than advertising uncertain VRAM as free.
- CLI smoke allocation requires explicit `--allocation-smoke`, is capped at 64 MiB, applies the
  normal hardware/runtime/process safety policy and defaults to at least 1 GiB of live VRAM
  headroom.

## Deterministic fault coverage

The hardware-independent fake-driver suite verifies:

- complete capability serialization and injected probe failure;
- reservation visibility before the backend allocation call;
- successful move/release ownership and zero residual accounting;
- zero-byte, negative-device, empty-tag and over-budget refusal before backend work;
- backend allocation error/exception rollback;
- malformed backend-handle cleanup;
- malformed-handle cleanup failure retaining the Governor charge;
- explicit release failure preserving a valid retryable allocation;
- successful release retry; and
- destructor cleanup failure leaving accounting fail-closed.

The cross-platform CLI suite verifies command help, duplicate/unknown/missing arguments, zero and
over-64-MiB refusal, reserve-without-allocation refusal, structured JSON errors and either a valid
native capability record or a clean unavailable-driver result. The live closed-source driver probe
is intentionally skipped under sanitizers; the deterministic backend remains fully instrumented.

## Validation host

- Windows 11 Pro build `26200`;
- AMD Ryzen 9 7900X, 12 cores / 24 threads;
- 32 GiB system RAM;
- NVIDIA GeForce RTX 4070 Super, 12,282 MiB reported by `nvidia-smi`;
- NVIDIA driver 610.88;
- WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`;
- CUDA Driver API encoded version 13,030 (`13.3`);
- `nvcuda.dll` and WSL `libcuda.so.1` available;
- no CUDA toolkit or `nvcc` available.

## Exact validation results

All compiler configurations treated project warnings as errors.

| Toolchain/configuration | Exact result |
|---|---:|
| GCC Release + real metadata opt-ins | 17/17 passed in 14.40 s |
| GCC Debug + real metadata opt-ins | 17/17 passed in 34.09 s |
| GCC ASan + UBSan, leak detection and halt-on-error | 17/17 passed in 88.98 s |
| Clang 18 Release | 13/13 passed in 5.39 s |
| GCC Debug + libstdc++ debug iterators/assertions | 13/13 passed in 9.01 s |
| MSVC Release x64 | 13/13 passed in 6.16 s |
| MSVC Debug x64 | 13/13 passed in 8.84 s |
| TSan focused native binaries | core, range-artifact and CUDA-driver tests passed |
| Python `compileall`, manifest and importer suites | passed; 1 Windows symlink case skipped |
| Windows real capability + 4 MiB allocation/release | passed |
| WSL real capability + 4 MiB allocation/release | passed |

TSan binaries used `TSAN_OPTIONS=halt_on_error=1` with `setarch x86_64 -R` to avoid the known WSL
address-layout conflict.

The ASan/UBSan CTest run skipped only the live CUDA driver call. A separate attempted WSL
ASan-to-`libcuda.so.1` call returned `CUDA_ERROR_OUT_OF_MEMORY`, and LeakSanitizer attributed
50,980 bytes to allocations inside the closed-source WSL driver during `cuInit`. Vastloom's
fake-backend allocation/cleanup implementation remained instrumented and passed. Non-sanitized
Windows and WSL binaries exercised the real driver and allocation successfully.

## Safety and performance statement

The real smoke allocation was 4 MiB and immediately released. It is not a VRAM bandwidth,
allocation-latency or model-throughput benchmark. No model weight, KV cache, activation or expert
was copied to the GPU. The command's 64 MiB ceiling is a diagnostic bound, not the reusable API's
capacity; production allocations remain constrained by the runtime-derived Governor hard limit
and live device reserve.

## Known limitations and next bottleneck

- No CUDA kernels, modules, streams, events or asynchronous copies exist.
- No pinned host allocation or RAM-to-VRAM tensor transfer exists.
- No CUDA output has been compared with a CPU token/logit/route fixture.
- The CLI smoke targets device ordinal 0; multi-GPU placement policy is not implemented.
- The allocator checks live VRAM at acquisition time but does not yet monitor device pressure over
  a long-lived kernel/transfer workload.
- CUDA device loss, reset and errors after future asynchronous work remain unimplemented.
- macOS/Metal, ROCm and Vulkan are separate roadmap tracks.

The next M8 correctness slice is a minimal device buffer copy plus one CUDA reference operation
whose output can be compared exactly with an existing CPU fixture. The project still needs the
separately authorized official layer-0 payload gate before any claim about DeepSeek checkpoint
execution changes.
