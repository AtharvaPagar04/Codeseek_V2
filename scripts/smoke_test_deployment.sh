#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <frontend-url> <api-url> <api-key>"
  exit 1
fi

FRONTEND_URL="${1%/}"
API_URL="${2%/}"
API_KEY="$3"

echo "[1/5] frontend reachability"
curl -fsSL "$FRONTEND_URL" >/dev/null

echo "[2/5] backend health"
curl -fsSL -H "Authorization: Bearer $API_KEY" "$API_URL/api/v1/health"
echo

echo "[3/5] backend metrics"
curl -fsSL -H "Authorization: Bearer $API_KEY" "$API_URL/api/v1/metrics" | head
echo

echo "[4/5] oauth session endpoint"
curl -fsSL "$API_URL/auth/me"
echo

echo "[5/5] security headers spot check"
curl -fsSI "$FRONTEND_URL" | sed -n '1,12p'
curl -fsSI "$API_URL/api/v1/health" -H "Authorization: Bearer $API_KEY" | sed -n '1,12p'
echo

echo "smoke test completed"
