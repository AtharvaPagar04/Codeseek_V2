#!/bin/bash
# scripts/perf_baseline.sh - Performance baseline runner wrapper for CodeSeek
# [ignoring loop detection]

set -e

# Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Verify backend virtual environment exists
if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
    echo -e "${RED}ERROR: Backend virtual environment not found at backend/.venv.${NC}"
    echo "Please set it up first:"
    echo "  cd backend && uv venv && uv pip install -r requirements.txt"
    exit 1
fi

# Run the python performance baseline script passing all arguments
PYTHONPATH="$ROOT_DIR/backend" "$ROOT_DIR/backend/.venv/bin/python" "$ROOT_DIR/scripts/perf_baseline.py" "$@"
