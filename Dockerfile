# Stage 1: Build dashboard
FROM node:20-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ .
RUN npm run build

# Stage 2: Python server
FROM python:3.12-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ .
COPY --from=dashboard /dashboard/dist ./static/
EXPOSE 8080
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
