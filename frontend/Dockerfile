# Multi-stage Dockerfile for React Frontend
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
ARG APP_REVISION=local
LABEL org.opencontainers.image.revision=$APP_REVISION
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
