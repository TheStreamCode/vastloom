# Two-minute demo: no model download required.
# Build first if needed:  pwsh scripts/build.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$candidates = @(
    (Join-Path $root "build\release\vastloom.exe"),
    (Join-Path $root "build\vastloom.exe")
)
$binary = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $binary) {
    Write-Error "No vastloom binary found. Build it first: pwsh scripts/build.ps1"
    exit 1
}

$fixture = Join-Path $root "examples\models\vastloom-tinychat-v1.sgtm"

Write-Host "== version ==" -ForegroundColor Cyan
& $binary version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n== hardware (no model touched) ==" -ForegroundColor Cyan
& $binary doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n== deterministic bundled fixture ==" -ForegroundColor Cyan
& $binary self-test $fixture
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host @"

Demo complete.

Optional next steps (need extra, public inputs; see docs/USAGE.md):
  - SmolLM2-135M (269 MB) serve + JSON object mode
  - bundled loopback server:  vastloom serve examples/models/vastloom-tinychat-v1.sgtm --port 8099
  - DeepSeek reduced-payload probes: import the pinned range artifacts first

No weights are downloaded by this script.
"@ -ForegroundColor Green
