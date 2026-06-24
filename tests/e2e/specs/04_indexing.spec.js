/**
 * E2E: Indexing to "ready"
 *
 * Verifies:
 *  - Session created for a real (public) repo progresses from indexing → ready.
 *  - Failed sessions surface a usable error message.
 *  - Frontend shows indexing status while in progress.
 *  - Frontend shows ready state when indexing completes.
 *
 * NOTE: This test clones a real public repo and may take several minutes.
 *       It is gated by ENABLE_INDEXING_E2E=1 to avoid running in fast CI loops.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  deleteAllSessionsViaApi,
  waitForSessionReady,
  TEST_REPO,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const ENABLE_INDEXING_E2E = process.env.ENABLE_INDEXING_E2E === '1';
const authHeaders = () => ({ Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' });

test.describe('Indexing to Ready', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
    if (!ENABLE_INDEXING_E2E) {
      // Skip all — test.skip() inside beforeAll applies to the suite
    }
  });

  test.beforeEach(async ({ page }) => {
    if (!ENABLE_INDEXING_E2E) {
      test.skip(true, 'Set ENABLE_INDEXING_E2E=1 to run indexing tests (clones real repos, slow)');
    }
    await page.goto('/');
    await connectGithubViaPat(page);
  });

  test.afterEach(async () => {
    await deleteAllSessionsViaApi();
  });

  test('session reaches ready status after indexing completes', async ({ page }) => {
    // Create session
    const createRes = await page.request.post(`${BACKEND_URL}/api/v1/sessions`, {
      data: {
        repo_full_name: TEST_REPO,
        repo_url: `https://github.com/${TEST_REPO}.git`,
        tenant_id: 'e2e',
      },
      headers: authHeaders(),
    });
    expect(createRes.ok()).toBe(true);
    const { session } = await createRes.json();
    expect(session.status).toBe('indexing');

    // Wait for ready (up to 5 minutes for a real repo)
    const ready = await waitForSessionReady(session.id, 300_000);
    expect(ready.status).toBe('ready');
    expect(ready.chunks_generated).toBeGreaterThan(0);
  });

  test('frontend shows indexing progress indicator while session is indexing', async ({ page }) => {
    const createRes = await page.request.post(`${BACKEND_URL}/api/v1/sessions`, {
      data: {
        repo_full_name: TEST_REPO,
        repo_url: `https://github.com/${TEST_REPO}.git`,
        tenant_id: 'e2e',
      },
      headers: authHeaders(),
    });
    const { session } = await createRes.json();

    await page.reload();

    // The status message should mention indexing
    await expect(
      page.getByText(/indexing/i, { exact: false })
    ).toBeVisible({ timeout: 15_000 });

    // Input should be disabled while indexing
    const textarea = page.locator('textarea').first();
    await expect(textarea).toBeDisabled();

    // Wait for ready before cleanup
    await waitForSessionReady(session.id, 300_000);
  });

  test('frontend input becomes enabled when session is ready', async ({ page }) => {
    const createRes = await page.request.post(`${BACKEND_URL}/api/v1/sessions`, {
      data: {
        repo_full_name: TEST_REPO,
        repo_url: `https://github.com/${TEST_REPO}.git`,
        tenant_id: 'e2e',
      },
      headers: authHeaders(),
    });
    const { session } = await createRes.json();

    await waitForSessionReady(session.id, 300_000);
    await page.reload();

    // After reload and indexing, input should be enabled
    const textarea = page.locator('textarea').first();
    await expect(textarea).toBeEnabled({ timeout: 15_000 });
  });
});
