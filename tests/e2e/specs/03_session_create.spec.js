/**
 * E2E: Session Creation + Deduplication
 *
 * Verifies:
 *  - Session can be created for a repo.
 *  - Duplicate create for the same repo returns the existing session (no duplicate rows).
 *  - Session appears in session list.
 *  - Session deletion removes it from the list.
 *  - Frontend session list updates after creation.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  listSessionsViaApi,
  deleteAllSessionsViaApi,
  TEST_REPO,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const authHeaders = () => ({ Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' });

async function createSessionViaApi(page, repoFullName = TEST_REPO) {
  const res = await page.request.post(`${BACKEND_URL}/api/v1/sessions`, {
    data: {
      repo_full_name: repoFullName,
      repo_url: `https://github.com/${repoFullName}.git`,
      tenant_id: 'e2e',
    },
    headers: authHeaders(),
  });
  if (!res.ok()) {
    const body = await res.json().catch(() => ({}));
    throw new Error(`Session create failed: ${res.status()} — ${body.detail || ''}`);
  }
  const data = await res.json();
  return data.session;
}

test.describe('Session Creation', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);
  });

  test.afterEach(async () => {
    await deleteAllSessionsViaApi();
  });

  test('create session returns a session object with status=indexing', async ({ page }) => {
    const session = await createSessionViaApi(page);

    expect(session).toHaveProperty('id');
    expect(session).toHaveProperty('status');
    expect(['indexing', 'ready']).toContain(session.status);
    expect(session.repo_full_name).toBe(TEST_REPO);
  });

  test('session appears in list after creation', async ({ page }) => {
    const session = await createSessionViaApi(page);
    const sessions = await listSessionsViaApi();
    const found = sessions.find((s) => s.id === session.id);
    expect(found).toBeTruthy();
  });

  test('duplicate create for same repo returns existing session (no duplicate)', async ({ page }) => {
    const first = await createSessionViaApi(page);
    const second = await createSessionViaApi(page);

    expect(first.id).toBe(second.id);

    const sessions = await listSessionsViaApi();
    const matches = sessions.filter((s) => s.repo_full_name === TEST_REPO);
    expect(matches.length).toBe(1);
  });

  test('delete session removes it from list', async ({ page }) => {
    const session = await createSessionViaApi(page);

    const deleteRes = await page.request.delete(
      `${BACKEND_URL}/api/v1/sessions/${session.id}`,
      { headers: authHeaders() }
    );
    expect(deleteRes.ok()).toBe(true);

    const sessions = await listSessionsViaApi();
    const found = sessions.find((s) => s.id === session.id);
    expect(found).toBeUndefined();
  });

  test('frontend sidebar shows new session after creation', async ({ page }) => {
    await createSessionViaApi(page);
    await page.reload();

    // The repo name should appear somewhere in the sidebar
    const repoName = TEST_REPO.split('/')[1];
    await expect(page.getByText(repoName, { exact: false })).toBeVisible({ timeout: 10_000 });
  });
});
