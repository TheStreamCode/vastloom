# Alpha 2 Windows validation report

Validation date: 2026-08-09

This report records a fresh validation of the imported `v0.2.0-alpha.2` source on
the Windows 11 reference PC. It supplements, rather than replaces,
`ALPHA2_VALIDATION.md`.

## Reference host

- CPU: AMD Ryzen 9 7900X, 24 logical CPUs;
- GPU: NVIDIA GeForce RTX 4070 Super, 12 GiB VRAM, driver 610.88;
- memory: 33,407,475,712 bytes total;
- storage: Lexar SSD NM790 2 TB NVMe on `C:`;
- Windows compiler: MSVC 19.44.35228.0;
- WSL distribution: Ubuntu 24.04.4 LTS;
- Linux compilers: GCC 13.3.0 and Clang 18.1.3.

No CUDA toolkit was installed, so this validation does not include CUDA
compilation or execution.

## Findings and fixes

The imported source initially failed MSVC with warnings treated as errors:

- SHA-256 padding passed an unsigned integer to a byte fill operation;
- the Windows CPU probe used the deprecated MSVC `getenv` interface;
- Clang diagnosed a big-endian byte-swap helper as unused on little-endian hosts.

The fills now use an explicit `std::uint8_t`, Windows uses `_dupenv_s`, and the
conditionally useful helper is marked `maybe_unused`. Both build scripts and all
Debug/Release presets now keep project warnings fatal. A separate, mutually
exclusive ThreadSanitizer build option was added.

## Test matrix

| Toolchain/configuration | Result |
|---|---:|
| MSVC 19.44 Release, `/WX` | 4/4 passed |
| MSVC 19.44 Debug, `/WX` | 4/4 passed |
| GCC 13 Release, `-Werror` | 4/4 passed |
| GCC 13 Debug, `-Werror` | 4/4 passed |
| Clang 18 Release, `-Werror` | 4/4 passed |
| GCC 13 Debug + ASan + UBSan | 4/4 passed |
| GCC 13 Debug + ThreadSanitizer | 4/4 passed |
| GCC 13 Debug + libstdc++ debug/assertions | 4/4 passed |

ASan used leak detection and halt-on-error; UBSan and TSan used halt-on-error.
GCC ThreadSanitizer under WSL can terminate at startup because of an incompatible
random memory mapping. The four TSan cases were therefore run with ASLR disabled
only for each test process using `setarch x86_64 -R`; no global Windows or WSL
security setting was changed.

## Manual smoke tests

The native MSVC Release executable passed `version`, `doctor --json`, manifest
verification, `model-info --json`, the five-vector `self-test --json`, and an
accented Italian one-shot chat. The chat returned
`La capitale d'Italia è Roma.` with `finish_reason=end_of_turn`.

## Scope boundary

This remains validation of the bundled synthetic TinyTransformer. No real
open-weight model, CUDA kernel, MoE execution, or NVMe expert streaming is claimed
by this report.
