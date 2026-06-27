# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_PATH=/api
ARG VITE_GRAFANA_URL=/grafana/
ARG VITE_MAX_UPLOAD_MB=10

ENV VITE_API_BASE_PATH=$VITE_API_BASE_PATH
ENV VITE_GRAFANA_URL=$VITE_GRAFANA_URL
ENV VITE_MAX_UPLOAD_MB=$VITE_MAX_UPLOAD_MB

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]