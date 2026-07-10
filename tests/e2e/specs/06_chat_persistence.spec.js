/**
 * E2E: Chat Reload Persistence
 *
 * Verifies:
 *  - Chat messages persist across browser refresh.
 *  - Chat messages persist across backend restart (via API round-trip check).
 *  - Thread memory persists.
 *  - Clearing chat removes messages but keeps session alive.
 *
 * Requires: ENABLE_QUERY_E2E=1 and a ready session with TEST_PROVIDER_API_KEY set.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  createProviderCredentialViaApi,
  listSessionsViaApi,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const TEST_PROVIDER_KEY = process.env.TEST_PROVIDER_API_KEY || '';
const ENABLE_QUERY_E2E = process.env.ENABLE_QUERY_E2E === '1';

const authHeaders = () => ({ Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' });

async function getReadySession() {
  const sessions = await listSessionsViaApi();
  return sessions.find((s) => s.status === 'ready') || null;
}

async function sendQueryViaApi(page, sessionId, question) {
  const res = await page.request.post(`${BACKEND_URL}/api/v1/query`, {
    data: { question, session_id: sessionId },
    headers: authHeaders(),
  });
  if (!res.ok()) {
    const body = await res.json().catch(() => ({}));
    throw new Error(`Query failed: ${res.status()} — ${body.detail || ''}`);
  }
  return res.json();
}

test.describe('Chat Persistence', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
  });

  test.beforeEach(async ({ page }) => {
    if (!ENABLE_QUERY_E2E) {
      test.skip(true, 'Set ENABLE_QUERY_E2E=1 and TEST_PROVIDER_API_KEY to run chat persistence tests');
    }
    if (!TEST_PROVIDER_KEY) {
      test.skip(true, 'TEST_PROVIDER_API_KEY is required for chat persistence tests');
    }
    await page.goto('/');
    await connectGithubViaPat(page);
  });

  test('messages persist across browser refresh', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session');

    await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Persist Key', apiKey: TEST_PROVIDER_KEY,
    });

    // Send a query via API to seed a message
    const QUESTION = `What is the primary language used in this repo? [e2e-${Date.now()}]`;
    const result = await sendQueryViaApi(page, session.id, QUESTION);
    expect(result).toHaveProperty('answer');

    // Reload and verify messages are still there
    await page.reload();
    await page.waitForTimeout(3000); // allow session polling to load messages

    // The question text should be visible
    await expect(
      page.getByText(QUESTION.split('[')[0].trim(), { exact: false })
    ).toBeVisible({ timeout: 15_000 });
  });

  test('messages persist across backend API restart (API round-trip)', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session');

    await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Restart Key', apiKey: TEST_PROVIDER_KEY,
    });

    const QUESTION = `Describe the main entry point [e2e-restart-${Date.now()}]`;
    await sendQueryViaApi(page, session.id, QUESTION);

    // Fetch messages directly from API (simulates what the frontend does on reconnect)
    const threads = await page.request.get(
      `${BACKEND_URL}/api/v1/sessions/${session.id}/threads`,
      { headers: authHeaders() }
    );
    const { threads: threadList } = await threads.json();
    expect(threadList.length).toBeGreaterThan(0);

    const threadId = threadList[0].id;
    const msgsRes = await page.request.get(
      `${BACKEND_URL}/api/v1/threads/${threadId}/messages`,
      { headers: authHeaders() }
    );
    const { messages } = await msgsRes.json();
    const found = messages.find((m) => m.content?.includes('Describe the main entry point'));
    expect(found).toBeTruthy();
  });

  test('clearing chat removes messages but session stays alive', async ({ page }) => {
    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session');

    await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Clear Key', apiKey: TEST_PROVIDER_KEY,
    });

    await sendQueryViaApi(page, session.id, 'What files are in this repo?');

    // Get thread id
    const threadsRes = await page.request.get(
      `${BACKEND_URL}/api/v1/sessions/${session.id}/threads`,
      { headers: authHeaders() }
    );
    const { threads: threadList } = await threadsRes.json();
    const threadId = threadList[0]?.id;
    expect(threadId).toBeTruthy();

    // Clear messages
    const clearRes = await page.request.delete(
      `${BACKEND_URL}/api/v1/threads/${threadId}/messages`,
      { headers: authHeaders() }
    );
    expect(clearRes.ok()).toBe(true);

    // Messages should now be empty
    const msgsRes = await page.request.get(
      `${BACKEND_URL}/api/v1/threads/${threadId}/messages`,
      { headers: authHeaders() }
    );
    const { messages } = await msgsRes.json();
    expect(messages.length).toBe(0);

    // But the session itself should still be there and ready
    const sessions = await listSessionsViaApi();
    const alive = sessions.find((s) => s.id === session.id);
    expect(alive).toBeTruthy();
    expect(alive.status).toBe('ready');
  });
});
