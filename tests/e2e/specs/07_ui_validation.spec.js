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

test.describe('UI Validation', () => {
  test.beforeAll(async () => {
    if (!await backendIsHealthy()) throw new Error('Backend unreachable');
  });

  test('mobile layout keeps session and composer usable', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');

    await expect(page.getByRole('button', { name: /toggle sidebar/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /connect github/i })).toBeVisible();
    await page.getByRole('button', { name: /toggle sidebar/i }).click();
    await expect(page.getByRole('button', { name: /\+ new session/i })).toBeVisible();
  });

  test('expired auth state is visible after logout', async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);
    await page.reload();

    const logoutRes = await page.request.post(`${BACKEND_URL}/auth/logout`);
    expect(logoutRes.ok()).toBe(true);

    await page.reload();
    await expect(page.getByText(/github session expired/i)).toBeVisible({ timeout: 15_000 });
  });

  test('query sources render in the UI for successful responses', async ({ page }) => {
    test.skip(!ENABLE_QUERY_E2E || !TEST_PROVIDER_KEY, 'Requires ENABLE_QUERY_E2E=1 and TEST_PROVIDER_API_KEY');

    await page.goto('/');
    await connectGithubViaPat(page);

    const session = await getReadySession();
    if (!session) test.skip(true, 'No ready session');

    await createProviderCredentialViaApi(page, {
      provider: 'groq',
      label: 'UI Source Key',
      apiKey: TEST_PROVIDER_KEY,
    });

    await page.reload();
    await page.waitForSelector('textarea', { timeout: 15_000 });
    await page.locator('textarea').first().fill('What does this repository do?');
    await page.locator('textarea').first().press('Enter');

    await expect(page.getByText(/sources/i, { exact: false })).toBeVisible({ timeout: 60_000 });
    await expect(page.locator('button[title="Copy path"]').first()).toBeVisible({ timeout: 60_000 });
  });

  test('missing provider guidance is visible in API config', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /api config/i }).click();
    await expect(page.getByText(/add a provider key before sending queries/i)).toBeVisible({ timeout: 10_000 });
  });
});
