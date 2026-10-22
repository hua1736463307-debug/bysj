ARG NODE_BASE=node:22-bookworm-slim@sha256:de170b39439eb0e5a3444ad7d60b015614a381e10fecae2e30ec9e735122f38d
ARG NGINX_BASE=nginx:1.27-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10
FROM ${NODE_BASE} AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
# Only lock-integrity-matched package tarballs are provided, no npm credentials.
RUN --mount=type=bind,from=npmcache,target=/npm-cache,readonly \
    npm ci --offline --cache=/npm-cache --logs-dir=/tmp/npm-logs --no-audit --no-fund
COPY frontend/ ./
ARG VITE_API_BASE=/api/v1
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm test && npm run build
FROM ${NGINX_BASE} AS runtime
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
