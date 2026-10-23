#!/usr/bin/env bash
# Two-minute demo: no model download required.
# Build first if needed:  scripts/build.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

binary=""
for candidate in "$root/build/release/vastloom" "$root/build/vastloom"; do
    if [[ -x "$candidate" ]]; then
        binary="$candidate"
        break
    fi
done
if [[ -z "$binary" ]]; then
    echo "No vastloom binary found. Build it first: scripts/build.sh" >&2
    exit 1
fi

fixture="$root/examples/models/vastloom-tinychat-v1.sgtm"

echo "== version =="
"$binary" version

echo
echo "== hardware (no model touched) =="
"$binary" doctor

echo
echo "== deterministic bundled fixture =="
"$binary" self-test "$fixture"

cat <<'EOF'

Demo complete.

Optional next steps (need extra, public inputs; see docs/USAGE.md):
  - SmolLM2-135M (269 MB) serve + JSON object mode
  - bundled loopback server:  vastloom serve examples/models/vastloom-tinychat-v1.sgtm --port 8099
  - DeepSeek reduced-payload probes: import the pinned range artifacts first

No weights are downloaded by this script.
EOF
