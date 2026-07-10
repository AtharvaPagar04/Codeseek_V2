# Codeseek E2E Tests

Playwright-based end-to-end test suite for Codeseek deployment verification.

## Prerequisites

- Node.js ≥ 18
- A running Codeseek backend + frontend (local or deployed)
- Playwright browsers installed: `npm run install-browsers`

## Setup

```bash
cd tests/e2e
npm install
npm run install-browsers
```

## Environment Variables

Create a `.env.e2e` file or export these before running:

| Variable | Required | Description |
|---|---|---|
| `FRONTEND_URL` | Yes | Frontend origin, e.g. `http://localhost:5173` |
| `BACKEND_URL` | Yes | Backend origin, e.g. `http://localhost:8000` |
| `CODESEEK_API_KEY` | Yes | Backend bearer token |
| `GITHUB_TEST_PAT` | Yes | GitHub PAT with `repo` scope (for auth tests) |
| `TEST_REPO` | No | Full repo name to index, default `octocat/Hello-World` |
| `TEST_PROVIDER_API_KEY` | For query tests | Valid LLM provider API key (e.g. Groq) |
| `ENABLE_INDEXING_E2E` | For indexing tests | Set `1` to enable slow indexing tests |
| `ENABLE_QUERY_E2E` | For query/chat tests | Set `1` to enable query + chat tests |

## Running Tests

```bash
# Fast tests only (GitHub connect, provider CRUD, session CRUD)
FRONTEND_URL=http://localhost:5173 \
BACKEND_URL=http://localhost:8000 \
CODESEEK_API_KEY=your-api-key \
GITHUB_TEST_PAT=ghp_your_pat \
npm test

# Full suite including indexing (slow — clones real repo)
ENABLE_INDEXING_E2E=1 \
ENABLE_QUERY_E2E=1 \
TEST_PROVIDER_API_KEY=gsk_your_groq_key \
npm test

# Headed mode for debugging
npm run test:headed

# Debug a specific spec
npx playwright test specs/01_github_connect.spec.js --debug
```

## Test Specs

| Spec | Gates | What it tests |
|---|---|---|
| `01_github_connect.spec.js` | None | PAT connect, /auth/me, logout, frontend shows username, repo list |
| `02_provider_credential.spec.js` | None | Add, list, persist, activate, delete, missing-cred error |
| `03_session_create.spec.js` | None | Create, dedup, list, delete, frontend sidebar |
| `04_indexing.spec.js` | `ENABLE_INDEXING_E2E=1` | Indexing → ready, frontend progress, input unlock |
| `05_query.spec.js` | `ENABLE_QUERY_E2E=1` | Query API shape, answer in UI, input cleared, no-cred error |
| `06_chat_persistence.spec.js` | `ENABLE_QUERY_E2E=1` | Messages persist refresh, API round-trip, clear-chat |
| `07_ui_validation.spec.js` | Partial query gating | Mobile shell, expired auth notice, source rendering, provider guidance |

## CI Integration

Add to your CI pipeline:

```yaml
- name: Run Codeseek E2E tests
  working-directory: tests/e2e
  env:
    FRONTEND_URL: ${{ secrets.FRONTEND_URL }}
    BACKEND_URL: ${{ secrets.BACKEND_URL }}
    CODESEEK_API_KEY: ${{ secrets.CODESEEK_API_KEY }}
    GITHUB_TEST_PAT: ${{ secrets.GITHUB_TEST_PAT }}
  run: |
    npm ci
    npm run install-browsers
    npm test
```
