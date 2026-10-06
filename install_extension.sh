#!/bin/bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/microhh-fork" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")" && pwd)"
MH="$(cd "$1" && pwd)"
cp "$ROOT"/include/*.h "$MH/include/"
cp "$ROOT"/src/*.cxx "$MH/src/"
cp "$ROOT"/src/*.cu "$MH/src/"
echo "Copied extension sources. Now apply the integration points in:"
echo "  $ROOT/patches/model_hook.patch"
echo "and provide periodic-z pressure + 3-D spectral forcing in your fork."
