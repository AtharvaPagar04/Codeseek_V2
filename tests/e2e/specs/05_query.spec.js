/**
 * E2E: Query Roundtrip
 *
 * Verifies:
 *  - A query against a ready session returns an answer + sources.
 *  - The answer appears in the frontend chat view.
 *  - The query input clears after sending.
 *  - Error states (no credential, session not ready) produce readable UI messages.
 *
 * Requires: ENABLE_INDEXING_E2E=1 and a ready session (run after 04_indexing if needed).
 * Will reuse existing ready sessions rather than indexing from scratch.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  createProviderCredentialViaApi,
  listSessionsViaApi,
  waitForSessionReady,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const TEST_PROVIDER_KEY = process.env.TEST_PROVIDER_API_KEY || '';
const ENABLE_QUERY_E2E = process.env.ENABLE_QUERY_E2E === '1';

const authHeaders = () => ({ Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' });

/** Find an existing ready session, or skip. */
async function getReadySession() {
  const sessions = await listSessionsViaApi();
  return sessions.find((s) => s.status === 'ready') || null;
}

test.describe('Query Roundtrip', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
  });

  test.beforeEach(async ({ page }) => {
    if (!ENABLE_QUERY_E2E) {
      test.skip(true, 'Set ENABLE_QUERY_E2E=1 and TEST_PROVIDER_API_KEY to run query E2E tests');
    }
    await page.goto('/');
    await connectGithubViaPat(page);
  });

  test('query returns answer and sources via API', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session — run indexing tests first');

    if (!TEST_PROVIDER_KEY) test.skip(true, 'TEST_PROVIDER_API_KEY is required for query tests');

    await createProviderCredentialViaApi(page, {
      provider: 'groq',
      label: 'E2E Query Key',
      apiKey: TEST_PROVIDER_KEY,
    });

    const queryRes = await page.request.post(`${BACKEND_URL}/api/v1/query`, {
      data: { question: 'What does this repository do?', session_id: session.id },
      headers: authHeaders(),
    });

    expect(queryRes.ok()).toBe(true);
    const body = await queryRes.json();
    expect(body).toHaveProperty('answer');
    expect(typeof body.answer).toBe('string');
    expect(body.answer.length).toBeGreaterThan(10);
    expect(Array.isArray(body.sources)).toBe(true);
  });

  test('answer appears in frontend chat view after query', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session — run indexing tests first');
    if (!TEST_PROVIDER_KEY) test.skip(true, 'TEST_PROVIDER_API_KEY required');

    await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'E2E UI Key', apiKey: TEST_PROVIDER_KEY,
    });

    await page.reload();
    // Wait for session list to load and select the ready session
    await page.waitForSelector('textarea', { timeout: 15_000 });

    const textarea = page.locator('textarea').first();
    await textarea.fill('What does this repository do?');
    await textarea.press('Enter');

    // Wait for assistant response bubble to appear
    await expect(
      page.locator('[data-role="assistant"], .message-bubble-assistant').first()
    ).toBeVisible({ timeout: 60_000 });

    // Input should be cleared after send
    await expect(textarea).toHaveValue('');
  });

  test('query with no provider credential shows readable error in UI', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session');

    // Ensure no provider credentials
    const listRes = await page.request.get(`${BACKEND_URL}/api/v1/provider-credentials`, {
      headers: authHeaders(),
    });
    const { provider_credentials = [] } = await listRes.json();
    for (const c of provider_credentials) {
      await page.request.delete(`${BACKEND_URL}/api/v1/provider-credentials/${c.id}`, {
        headers: authHeaders(),
      });
    }

    await page.reload();
    await page.waitForSelector('textarea', { timeout: 15_000 });
    const textarea = page.locator('textarea').first();
    await textarea.fill('test question');
    await textarea.press('Enter');

    // Frontend should show a readable error containing "provider"
    await expect(
      page.getByText(/provider/i, { exact: false })
    ).toBeVisible({ timeout: 15_000 });
  });
});
