FROM node:20.19.0-alpine AS build

LABEL maintainer="CodeSeek Team"
LABEL org.opencontainers.image.title="CodeSeek Frontend"
LABEL org.opencontainers.image.description="Production build stage for CodeSeek V2 Frontend"

WORKDIR /app

COPY package*.json /app/
RUN npm ci

COPY . /app

ARG VITE_API_BASE_URL
ARG VITE_API_KEY
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_API_KEY=${VITE_API_KEY}

RUN npm run build

FROM nginx:1.27.5-alpine

LABEL maintainer="CodeSeek Team"
LABEL org.opencontainers.image.title="CodeSeek Frontend"
LABEL org.opencontainers.image.description="Production web server stage for CodeSeek V2 Frontend"
LABEL stage="production"

COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# 1. Explicitly define PORT so envsubst can use it, defaulting to 80 for local dev
ENV PORT=80
ENV ALLOWED_HOST=_

EXPOSE ${PORT}

# 2. Updated healthcheck to use the dynamic ${PORT} variable
# Note: Using shell form (not exec array) ensures ${PORT} is interpolated
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:${PORT}/healthz || exit 1