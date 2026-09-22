FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY . .
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine
RUN apk add --no-cache openssl
COPY --from=build /app/dist /usr/share/nginx/html

# ── Release identification (ISO 9001 8.5.2 traceability) ──────────────────────
# The served assets are static, so the build identity is written into the web
# root as /build.json rather than read from the environment at runtime. Fetch it
# with: curl -ks https://<host>:3443/build.json
ARG BUILD_VERSION=0.0.0-dev
ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
RUN printf '{"app":"echomind-frontend","version":"%s","commit":"%s","date":"%s"}\n' \
      "$BUILD_VERSION" "$BUILD_COMMIT" "$BUILD_DATE" \
      > /usr/share/nginx/html/build.json

COPY nginx.conf /etc/nginx/conf.d/default.conf
# Self-signed cert for HTTPS without a domain (browser will show warning; accept to continue)
RUN mkdir -p /etc/nginx/ssl && \
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/key.pem \
    -out /etc/nginx/ssl/cert.pem \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
EXPOSE 80 443
