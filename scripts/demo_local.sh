#!/bin/bash
# demo_local.sh - Local demo helper script for CodeSeek
# Checks dependencies, infrastructure, and guides users on how to run CodeSeek locally.

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

# Modes
CHECK_ONLY=false
DRY_RUN=false

show_help() {
    echo -e "${BLUE}CodeSeek Local Demo Helper${NC}"
    echo "Usage: ./scripts/demo_local.sh [options]"
    echo ""
    echo "Options:"
    echo "  -c, --check-only    Perform status/dependency checks and exit immediately"
    echo "  -d, --dry-run       Print check descriptions and commands without executing network checks"
    echo "  -h, --help          Show this help message"
    echo ""
}

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--check-only) CHECK_ONLY=true ;;
        -d|--dry-run) DRY_RUN=true ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
    shift
done

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                  CodeSeek Local Demo Helper                          ${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

errors=0
warnings=0

# 1. Check Backend Virtual Environment
echo -n "Checking Backend Python Virtual Environment... "
if [ -d "$ROOT_DIR/backend/.venv" ]; then
    echo -e "${GREEN}OK${NC} (found at backend/.venv)"
else
    echo -e "${RED}MISSING${NC}"
    echo -e "  -> Please set up the backend virtual environment:"
    echo -e "     cd backend && uv venv && uv pip install -r requirements.txt"
    errors=$((errors + 1))
fi

# 2. Check Frontend Node Modules
echo -n "Checking Frontend Node Modules... "
if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo -e "${GREEN}OK${NC} (found at frontend/node_modules)"
else
    echo -e "${RED}MISSING${NC}"
    echo -e "  -> Please install frontend dependencies:"
    echo -e "     cd frontend && npm install"
    errors=$((errors + 1))
fi

# 3. Check environment configuration file (.env)
echo -n "Checking Backend environment configuration (.env)... "
if [ -f "$ROOT_DIR/backend/.env" ]; then
    echo -e "${GREEN}OK${NC} (found at backend/.env)"
else
    echo -e "${YELLOW}WARNING${NC} (backend/.env not found)"
    echo -e "  -> Using default environment. To customize, copy the example:"
    echo -e "     cp backend/.env.example backend/.env"
    warnings=$((warnings + 1))
fi

# 4. Check Database Configuration & Status
echo -n "Checking Database Configuration... "
db_backend="sqlite"
db_url="sqlite:///codeseek.db"
if [ -f "$ROOT_DIR/backend/.env" ]; then
    env_backend=$(grep -E "^CODESEEK_DB_BACKEND=" "$ROOT_DIR/backend/.env" | cut -d'=' -f2 | tr -d '"'\'' ')
    env_url=$(grep -E "^CODESEEK_DATABASE_URL=" "$ROOT_DIR/backend/.env" | cut -d'=' -f2 | tr -d '"'\'' ')
    if [ -n "$env_backend" ]; then
        db_backend="$env_backend"
    fi
    if [ -n "$env_url" ]; then
        db_url="$env_url"
    fi
fi

echo -e "${GREEN}OK${NC} (backend: ${db_backend})"
if [ "$db_backend" = "postgres" ]; then
    echo -e "  -> Configured with Postgres database URL: ${db_url}"
else
    echo -e "  -> Configured with SQLite database: backend/codeseek.db"
fi

# 5. Check Qdrant Vector DB Service
echo -n "Checking Qdrant service availability... "
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}SKIPPED${NC} (dry-run mode)"
else
    if command -v curl >/dev/null 2>&1; then
        qdrant_url="http://localhost:6333"
        if [ -f "$ROOT_DIR/backend/.env" ]; then
            q_host=$(grep -E "^QDRANT_HOST=" "$ROOT_DIR/backend/.env" | cut -d'=' -f2 | tr -d '"'\'' ')
            q_port=$(grep -E "^QDRANT_PORT=" "$ROOT_DIR/backend/.env" | cut -d'=' -f2 | tr -d '"'\'' ')
            if [ -n "$q_host" ] && [ -n "$q_port" ]; then
                qdrant_url="http://${q_host}:${q_port}"
            fi
        fi
        
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$qdrant_url/healthz" || true)
        if [ "$http_code" = "200" ]; then
            echo -e "${GREEN}OK${NC} (Qdrant active at $qdrant_url)"
        else
            echo -e "${RED}UNREACHABLE${NC} (HTTP $http_code at $qdrant_url)"
            echo -e "  -> Please start Qdrant before running CodeSeek:"
            echo -e "     docker compose up -d qdrant"
            errors=$((errors + 1))
        fi
    else
        echo -e "${YELLOW}UNKNOWN${NC} (curl not installed)"
        warnings=$((warnings + 1))
    fi
fi

echo ""
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo -e "Check Summary: ${GREEN}${errors} errors${NC}, ${YELLOW}${warnings} warnings${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"
echo ""

if [ "$CHECK_ONLY" = "true" ]; then
    if [ $errors -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
fi

if [ $errors -gt 0 ]; then
    echo -e "${RED}Please resolve the errors above before running the local demo.${NC}"
    exit 1
fi

# Print Startup Instructions
echo -e "${GREEN}All checks passed! CodeSeek is ready to run.${NC}"
echo ""
echo -e "${BLUE}Key Environment Variables of Note:${NC}"
echo "  - CODESEEK_ENABLE_INCREMENTAL_REINDEX=true    Enable experimental incremental reindexing"
echo "  - CODESEEK_API_KEY                            API secret bearer token for API security"
echo "  - CODESEEK_DB_BACKEND                         Database driver ('sqlite' or 'postgres')"
echo "  - OLLAMA_HOST / OLLAMA_MODEL                  Local LLM provider variables (optional)"
echo ""
echo -e "${BLUE}Startup Commands:${NC}"
echo -e "  1. Start the FastAPI Backend:"
echo -e "     ${YELLOW}cd backend && ./scripts/run_local_backend.sh${NC}"
echo ""
echo -e "  2. Start the React Frontend:"
echo -e "     ${YELLOW}cd frontend && npm run dev${NC}"
echo ""
echo -e "${BLUE}Local Demo URLs:${NC}"
echo -e "  - Frontend Application:  ${GREEN}http://localhost:5173${NC}"
echo -e "  - Backend API Gateway:   ${GREEN}http://localhost:8000${NC}"
echo -e "  - Backend Health Status: ${GREEN}http://localhost:8000/api/v1/health${NC}"
echo ""
echo -e "Enjoy demoing CodeSeek!"
