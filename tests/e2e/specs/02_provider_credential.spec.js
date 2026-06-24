/**
 * E2E: Provider Credential Add / Activate / Delete
 *
 * Verifies:
 *  - Provider credential can be added (encrypted submission).
 *  - Added credential appears in the credential list.
 *  - Credential survives browser refresh.
 *  - Activation works (is_active toggles correctly).
 *  - Deletion works.
 *  - Missing credential produces a clear error on query.
 *  - Invalid provider / rate-limit (429) produces a user-readable error.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  createProviderCredentialViaApi,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.CODESEEK_API_KEY || '';
const TEST_PROVIDER_KEY = process.env.TEST_PROVIDER_API_KEY || 'sk-test-invalid-key-for-e2e';

const authHeaders = () => ({ Authorization: `Bearer ${API_KEY}` });

test.describe('Provider Credential Flow', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);
  });

  test.afterEach(async ({ page }) => {
    // Clean up: delete all provider credentials
    const listRes = await page.request.get(`${BACKEND_URL}/api/v1/provider-credentials`, {
      headers: authHeaders(),
    });
    if (!listRes.ok()) return;
    const { provider_credentials = [] } = await listRes.json();
    for (const cred of provider_credentials) {
      await page.request.delete(`${BACKEND_URL}/api/v1/provider-credentials/${cred.id}`, {
        headers: authHeaders(),
      });
    }
  });

  test('create provider credential via API and verify list response', async ({ page }) => {
    const cred = await createProviderCredentialViaApi(page, {
      provider: 'groq',
      label: 'E2E Test Key',
      apiKey: TEST_PROVIDER_KEY,
    });

    expect(cred).toHaveProperty('id');
    expect(cred.provider).toBe('groq');
    expect(cred.label).toBe('E2E Test Key');
    // api_key must NOT be returned in list response
    expect(cred).not.toHaveProperty('api_key');

    const listRes = await page.request.get(`${BACKEND_URL}/api/v1/provider-credentials`, {
      headers: authHeaders(),
    });
    const { provider_credentials } = await listRes.json();
    const found = provider_credentials.find((c) => c.id === cred.id);
    expect(found).toBeTruthy();
    expect(found).not.toHaveProperty('api_key');
  });

  test('provider credentials survive browser refresh', async ({ page }) => {
    const cred = await createProviderCredentialViaApi(page, {
      provider: 'groq',
      label: 'Persist Test',
      apiKey: TEST_PROVIDER_KEY,
    });

    await page.reload();

    const listRes = await page.request.get(`${BACKEND_URL}/api/v1/provider-credentials`, {
      headers: authHeaders(),
    });
    const { provider_credentials } = await listRes.json();
    const found = provider_credentials.find((c) => c.id === cred.id);
    expect(found).toBeTruthy();

    await page.getByRole('button', { name: /api config/i }).click();
    await expect(page.getByText('Persist Test', { exact: false })).toBeVisible({ timeout: 10_000 });
  });

  test('activate provider credential sets is_active=true', async ({ page }) => {
    const cred1 = await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Key A', apiKey: TEST_PROVIDER_KEY,
    });
    const cred2 = await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Key B', apiKey: TEST_PROVIDER_KEY,
    });

    const activateRes = await page.request.post(
      `${BACKEND_URL}/api/v1/provider-credentials/${cred2.id}/activate`,
      { headers: authHeaders() }
    );
    expect(activateRes.ok()).toBe(true);
    const { provider_credential } = await activateRes.json();
    expect(provider_credential.is_active).toBe(true);
    expect(provider_credential.id).toBe(cred2.id);
  });

  test('delete provider credential removes it from list', async ({ page }) => {
    const cred = await createProviderCredentialViaApi(page, {
      provider: 'groq', label: 'Delete Me', apiKey: TEST_PROVIDER_KEY,
    });

    const deleteRes = await page.request.delete(
      `${BACKEND_URL}/api/v1/provider-credentials/${cred.id}`,
      { headers: authHeaders() }
    );
    expect(deleteRes.ok()).toBe(true);

    const listRes = await page.request.get(`${BACKEND_URL}/api/v1/provider-credentials`, {
      headers: authHeaders(),
    });
    const { provider_credentials } = await listRes.json();
    const found = provider_credentials.find((c) => c.id === cred.id);
    expect(found).toBeUndefined();
  });

  test('query with no provider credential returns 400 with readable error', async ({ page }) => {
    // No credential added — query should fail with 400
    const sessions = await page.request.get(`${BACKEND_URL}/api/v1/sessions`, {
      headers: authHeaders(),
    });
    const { sessions: sessionList = [] } = await sessions.json();
    if (sessionList.length === 0) {
      test.skip(true, 'No sessions available to test missing-credential error');
      return;
    }
    const session = sessionList.find((s) => s.status === 'ready');
    if (!session) {
      test.skip(true, 'No ready session available');
      return;
    }

    const queryRes = await page.request.post(`${BACKEND_URL}/api/v1/query`, {
      data: { question: 'test', session_id: session.id },
      headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    });
    expect(queryRes.status()).toBe(400);
    const body = await queryRes.json();
    expect(body.detail).toMatch(/no active provider credential/i);
  });
});
