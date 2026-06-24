import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Codeseek E2E tests.
 *
 * Required environment variables (set in .env.e2e or exported):
 *   FRONTEND_URL        Frontend origin, e.g. http://localhost:5173
 *   BACKEND_URL         Backend origin, e.g. http://localhost:8000
 *   CODESEEK_API_KEY    Backend bearer token
 *   GITHUB_TEST_PAT     GitHub PAT with repo scope (for E2E tests)
 *   TEST_REPO           Full repo name to use in tests, e.g. "octocat/hello-world"
 */

export default defineConfig({
  testDir: './specs',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,           // Run sequentially — shared backend state
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report', open: 'never' }]],

  use: {
    baseURL: process.env.FRONTEND_URL || 'http://localhost:5173',
    headless: true,
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
