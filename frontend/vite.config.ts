import path from 'path';
import fs from 'fs';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

const useHttps = process.env.VITE_DEV_HTTPS === '1';
const certPath = process.env.VITE_SSL_CERT;
const keyPath = process.env.VITE_SSL_KEY;

/** Trusted HTTPS: use mkcert certs when set (no browser warning). Else use basic-ssl (self-signed) when VITE_DEV_HTTPS=1. */
function getHttpsConfig(): { https: { cert: Buffer; key: Buffer } } | undefined {
  if (!useHttps) return undefined;
  if (certPath && keyPath) {
    try {
      const cert = fs.readFileSync(path.resolve(certPath));
      const key = fs.readFileSync(path.resolve(keyPath));
      return { https: { cert, key } };
    } catch (e) {
      console.warn('VITE_SSL_CERT/VITE_SSL_KEY could not be read, falling back to self-signed:', (e as Error).message);
    }
  }
  // basicSsl plugin handles self-signed HTTPS when no cert files are provided.
  return undefined;
}

const httpsConfig = getHttpsConfig();

/** Match docker-compose voice host port (default 8002); set VOICE_HOST_PORT when running Vite against compose. */
const voiceDevPort = process.env.VOICE_HOST_PORT || '8002';

export default defineConfig(() => ({
  server: {
    port: 3000,
    host: '0.0.0.0',
    ...(httpsConfig || {}),
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
      '/voice': { target: `http://127.0.0.1:${voiceDevPort}`, changeOrigin: true, ws: true, rewrite: (p) => p.replace(/^\/voice/, '') },
    },
  },
  plugins: useHttps ? [react(), basicSsl()] : [react()],
  resolve: { alias: { '@': path.resolve(__dirname, '.') } },
}));
