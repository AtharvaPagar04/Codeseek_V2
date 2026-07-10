/**
 * E2E: GitHub Connect (PAT flow)
 *
 * Verifies:
 *  - GitHub PAT connect via encrypted submission sets an auth session.
 *  - /auth/me returns the authenticated user after connect.
 *  - Logout clears the session.
 *  - Frontend shows the username after connect.
 */

import { test, expect } from '@playwright/test';
import {
  backendIsHealthy,
  connectGithubViaPat,
  TEST_REPO,
} from '../helpers/api.js';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

test.describe('GitHub Connect — PAT flow', () => {
  test.beforeAll(async () => {
    const healthy = await backendIsHealthy();
    if (!healthy) {
      throw new Error(`Backend at ${BACKEND_URL} is not reachable. Start the backend before running E2E tests.`);
    }
  });

  test('POST /auth/github/token connects GitHub and returns username', async ({ page }) => {
    await page.goto('/');
    const result = await connectGithubViaPat(page);

    expect(result).toHaveProperty('username');
    expect(typeof result.username).toBe('string');
    expect(result.username.length).toBeGreaterThan(0);
  });

  test('/auth/me returns authenticated=true after PAT connect', async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);

    const res = await page.request.get(`${BACKEND_URL}/auth/me`);
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.authenticated).toBe(true);
    expect(body.user?.username).toBeTruthy();
  });

  test('frontend StatusBar shows username after GitHub connect', async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);
    await page.reload();

    // Username text should appear in the header area
    const username = await page.request
      .get(`${BACKEND_URL}/auth/me`)
      .then((r) => r.json())
      .then((d) => d.user?.username);

    await expect(page.getByText(username, { exact: false })).toBeVisible({ timeout: 10_000 });
  });

  test('repo picker lists repositories for the authenticated user', async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);
    await page.reload();

    await page.getByRole('button', { name: /\+ new session/i }).click();
    await expect(page.getByPlaceholder(/filter repositories/i)).toBeVisible({ timeout: 15_000 });

    const repoName = TEST_REPO.split('/')[1];
    await expect(page.getByText(repoName, { exact: false })).toBeVisible({ timeout: 15_000 });
  });

  test('POST /auth/logout clears the session', async ({ page }) => {
    await page.goto('/');
    await connectGithubViaPat(page);

    const logoutRes = await page.request.post(`${BACKEND_URL}/auth/logout`);
    expect(logoutRes.ok()).toBe(true);
    const logoutBody = await logoutRes.json();
    expect(logoutBody.logged_out).toBe(true);

    // /auth/me should now return 401 Unauthorized
    const meRes = await page.request.get(`${BACKEND_URL}/auth/me`);
    expect(meRes.status()).toBe(401);
  });

  test('frontend shows "Connect GitHub" button when logged out', async ({ page }) => {
    // Logout first
    await page.request.post(`${BACKEND_URL}/auth/logout`).catch(() => {});
    await page.goto('/');

    await expect(page.getByRole('button', { name: /connect github/i })).toBeVisible({ timeout: 10_000 });
  });
});
