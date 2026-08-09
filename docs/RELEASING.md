# Release process

Only maintainers with repository release access may publish Vastloom. Alpha releases use semantic
versioning with a prerelease suffix, for example `0.4.0-alpha.4`.

1. Confirm the target scope and compatibility impact. Update `include/vastloom/version.hpp`,
   `CHANGELOG.md`, `CITATION.cff` and CPack metadata together.
2. Build from a clean checkout with warnings as errors on Windows and Linux. Run Release, Debug,
   ASan/UBSan, Python offline tests and every applicable real-model parity test.
3. Regenerate bundled artifacts from pinned inputs. Record their size, SHA-256 and reproduction
   command; never add downloaded third-party checkpoints.
4. Run `cmake --install` into an empty staging directory and smoke-test the installed executable,
   verified fixture and exported `Vastloom::Core` CMake package.
5. Produce packages with CPack. Inspect package contents and compute SHA-256 checksums.
6. Add a validation report under `docs/reports/` containing exact results, performance, limitations,
   the tested commit and hardware. Do not promote simulated or warm-cache numbers as physical I/O.
7. Create a signed or annotated `vX.Y.Z` tag only after CI and CodeQL pass on the release commit.
8. Publish release notes from the changelog, attach packages and checksums, then verify the public
   tag, artifacts and documentation independently.

The tag workflow builds Windows and Linux packages as an additional reproducibility check. A green
workflow is necessary evidence, not a substitute for the local real-model and hardware validation.
