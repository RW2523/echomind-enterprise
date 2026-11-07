import { defineConfig, devices } from '@playwright/test';

/**
 * EchoMind Law end-to-end tests (frontend/e2e). Starts a fresh, demo-seeded Law dev backend on
 * 127.0.0.1:8110 and the Vite dev server on :5174; nothing runs against shared or production data.
 *   cd frontend && npx playwright test
 * With the demo running on 8110, choose other ports: E2E_API_PORT=8141 E2E_WEB_PORT=5184 npx playwright test
 */
const API = process.env.E2E_API_PORT ?? '8110';
const WEB = process.env.E2E_WEB_PORT ?? '5174';
const ports = { LAW_BACKEND_PORT: API, LAW_FRONTEND_PORT: WEB };

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,                       // an authenticator code can be used once; never retry blindly
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${WEB}`, trace: 'off', ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 },
    // S1.8: a synthetic microphone (a tone), so recording journeys run without hardware or prompts.
    permissions: ['microphone'],
    launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] },
  },
  webServer: [
    { command: '../scripts/law/e2e_backend.sh', url: `http://127.0.0.1:${API}/health`, reuseExistingServer: false, timeout: 120_000, env: ports },
    { command: '../scripts/law/dev_frontend.sh', url: `http://localhost:${WEB}`, reuseExistingServer: false, timeout: 120_000,
      env: { LAW_FRONTEND_HOST: 'localhost', ...ports } },
  ],
});
