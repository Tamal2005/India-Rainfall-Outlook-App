# ---- 1. build the React app --------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. API + static frontend in one image ------------------------------------
FROM python:3.12-slim
# curl is used by the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY --from=web /web/dist frontend/dist
# Model + data artifacts (data/models, data/prepped, data/disaggregation_knnr).
COPY data/ data/

ENV DATA_ROOT=/app/data \
    FRONTEND_DIST=/app/frontend/dist \
    SHOW_PROGRESS=0 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
USER app
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# One worker on purpose: every worker would hold its own copy of the models and
# KNNR library in memory. Scale with more containers instead.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
