#!/usr/bin/env bash
# Root-level convenience wrapper; delegates to backend/scripts/run_local_backend.sh.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"
exec ./scripts/run_local_backend.sh "$@"
