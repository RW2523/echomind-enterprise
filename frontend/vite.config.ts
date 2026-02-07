import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

const useHttps = process.env.VITE_DEV_HTTPS === '1';

export default defineConfig(() => ({
  server: {
    port: 3000,
    host: '0.0.0.0',
    ...(useHttps ? { https: true } : {}),
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
      '/voice': { target: 'http://127.0.0.1:8001', changeOrigin: true, ws: true, rewrite: (p) => p.replace(/^\/voice/, '') },
    },
  },
  plugins: useHttps ? [react(), basicSsl()] : [react()],
  resolve: { alias: { '@': path.resolve(__dirname, '.') } },
}));
