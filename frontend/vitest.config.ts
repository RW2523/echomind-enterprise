import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Unit and interaction tests (EchoMind Law S1.0). Kept separate from vite.config.ts so the dev
// server's proxy/HTTPS settings never affect test runs.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['**/*.test.{ts,tsx}'],
    exclude: ['node_modules/**', 'dist/**'],
    restoreMocks: true,
  },
});
