import { defineConfig, devices } from '@playwright/test';

/**
 * T14 legacy smoke (S1.12): the STANDARD edition's interface against a fresh standard-edition
 * backend on 127.0.0.1:8130 (scripts/law/legacy_backend.sh) and Vite on :5175.
 *   cd frontend && npx playwright test -c playwright.legacy.config.ts
 */
export default defineConfig({
  testDir: './e2e-legacy',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 240_000,
  reporter: [['list']],
  use: { baseURL: 'http://localhost:5175', trace: 'off', ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
  webServer: [
    { command: '../scripts/law/legacy_backend.sh', url: 'http://127.0.0.1:8130/health', reuseExistingServer: false, timeout: 120_000 },
    { command: '../scripts/law/dev_frontend.sh', url: 'http://localhost:5175', reuseExistingServer: false, timeout: 120_000,
      env: { LAW_FRONTEND_HOST: 'localhost', LAW_FRONTEND_PORT: '5175', LAW_API_TARGET: 'http://127.0.0.1:8130' } },
  ],
});
