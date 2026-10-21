# D2b.1 — device expert residency with bounded CPU fallback

Validation date: 2026-10-21. Track: DeepSeek V4 Flash 0731, `UD-IQ2_XXS`. Source: working tree
after the D1/D2a commits-to-be. This report covers the **first half of D2b** only (device
residency and fallback). The independent logits oracle (D2b.2) is not yet available; no parity
claim is made here.

## Scope

Exercise the existing CUDA matrix accelerator on the D2a payload path:

- keep routed-expert matrices resident in VRAM, bounded by a fixed ceiling;
- leave always-active trunk/global matrices on the CPU (`experts_only`);
- fall back to the CPU per matrix when the device cannot take it;
- prove the CPU and device runs produce the same layer-0 logits.

This is **not** a VRAM payload tier: expert bundles are still loaded into governed RAM first, and
the device holds copies of their matrices. `ResidencyObject` VRAM transitions remain unused on
this path.

## Changes

- `include/vastloom/cuda/cuda_matvec_accelerator.hpp`,
  `src/cuda/cuda_matvec_accelerator.cpp`: new `experts_only` option and `declined_recurring`
  statistic. When set, recurring trunk/global matrices are declined before any device work; the
  ceiling comment now matches the implementation (evict least-recently-used, then decline).
- `src/cli/main.cpp` (`deepseek-layer-probe`): `--cuda-experts`, `--gpu-limit MIB`,
  `--gpu-reserve MIB`, `--cpu-only`; the accelerator is attached as `graph.accelerator`; the JSON
  gains an `accelerator` object (`offered`, `accelerated`, `declined`, `declined_recurring`,
  `evicted`, `resident_matrices`, `resident_bytes`, `uploaded_bytes`, `kernel_ms`, `upload_ms`,
  `declined_by_type`). Default path unchanged (CPU-only, no accelerator key).
- `--dump-logits`, `--dump-hidden`, `--dump-trace` write headerless little-endian float32 vectors
  and a small JSON trace for the D2b.2 oracle.
- `tests/test_cuda_driver.cpp`: always-on fake-backend tests for the accelerator refusals
  (`experts_only` recurring decline, zero identity, unsupported type, ceiling below one matrix).
- `tests/test_deepseek_gguf_contract.py` + `CMakeLists.txt`: opt-in
  `vastloom_deepseek_v4_gpu_residency` contract. It runs the CPU probe and two GPU probes (16 MiB
  and 1 MiB ceilings), compares the full 129,280-logit vectors (declared `atol = 1e-3`), checks
  the accelerator counters and skips with exit 75 when no CUDA driver is present.

## Results (reference PC, RTX 4070 Super, 12 GiB)

Artifacts: metadata store plus trunk 138,867,160 B / 21 ranges, experts 45,219,840 B / 18 ranges,
globals 596,000,788 B / 6 ranges. Token 0, layer 0.

| Run | Accelerator counters | Logits vs CPU |
|---|---|---|
| CPU reference (default) | no accelerator key | — |
| `--cuda-experts --gpu-limit 16` | offered 38, accelerated 18, declined 20 (all recurring), evicted 12, resident 6 matrices / 15,230,976 B, uploaded 45,219,840 B, kernel 138.961 ms, upload 6.2527 ms | **129,280 values, max abs diff 0.0** (bit-identical) |
| `--cuda-experts --gpu-limit 1` | offered 38, accelerated 0, declined 38 (20 recurring + 18 ceiling), resident 0 B | **max abs diff 0.0** |

Both GPU runs report the same summary as the CPU run: logits 129,280, min −16.1273, max 15.5206,
argmax 30,528, experts 254/222/245/200/53/35, expert bytes 45,219,840, trunk 138,867,160,
globals 596,000,788.

The opt-in contract test passes with the same numbers (`accelerated=18`, `evicted=12`,
`resident=15,230,976 B`, fallback logits identical within `1e-3`).

## Gates

| Gate | Result |
|---|---|
| Release build (MSVC, warnings as errors) | PASS |
| Debug build (MSVC, warnings as errors) | PASS |
| `ctest --test-dir build\debug` | 15 tests: 14 passed, 0 failed, 1 skipped (`vastloom_ggml_compute`, ggml not configured) |
| `vastloom_cuda_driver_test` | PASS, including the new accelerator refusal cases |
| Python `compileall` / `ruff` | PASS (after the D2b.2 oracle file is finalized; see the D3 report for the final gate) |

## Not established

- **No independent logits oracle.** The CPU/GPU equality is a self-consistency check, not an
  external reference; D2b.2 must reproduce the CPU logits from the artifact bytes.
- **Not a VRAM payload tier.** The expert bytes are read into RAM and uploaded; `ResidencyObject`
  VRAM transitions have no caller here. The memory contract's "experts in VRAM with RAM fallback"
  is not met by this mechanism alone.
- No throughput claim: the GPU tier is opt-in and was not benchmarked as a product path; a slower
  device result would be reported, not hidden.
- One layer of 43; no full-model, llama.cpp or parity claim.

## Reproduction

```powershell
$V = "build\release\vastloom.exe"
$M = "<store>\models\original\deepseek-v4-flash-0731-iq2xxs-fbbb5b93--metadata"
$O = "<store>\models\optimized"
& $V deepseek-layer-probe $M <trunk> <experts> <globals> --token 0 --json `
  --dump-logits layer0-logits-cpu.f32 --dump-hidden layer0-hidden-cpu.f32
& $V deepseek-layer-probe $M <trunk> <experts> <globals> --token 0 --json `
  --cuda-experts --gpu-limit 16 --dump-logits layer0-logits-gpu16.f32
& $V deepseek-layer-probe $M <trunk> <experts> <globals> --token 0 --json `
  --cuda-experts --gpu-limit 1 --dump-logits layer0-logits-gpu1.f32
```

Artifact and dump paths are local validation data under `build/`; they are not committed.
