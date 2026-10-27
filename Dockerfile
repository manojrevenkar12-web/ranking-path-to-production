# One image for the pipeline components (Vertex AI runs it as the KFP base image), the deploy workflow, the relay
# and the console/API. The command decides which: `ptp pipeline ...`, `ptp serve relay`, `ptp serve api`.

FROM node:22-slim AS console
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PTP_ROOT=/app PTP_WEB=/app/web/dist
WORKDIR /app
RUN useradd --create-home --uid 10001 ptp
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[gcp]"
COPY config/ config/
COPY pipelines/ pipelines/
COPY ad_hoc/ ad_hoc/
COPY .github/workflows/ .github/workflows/
COPY --from=console /web/dist web/dist
USER 10001
EXPOSE 8000 8090
ENTRYPOINT ["ptp"]
CMD ["serve", "api", "--host", "0.0.0.0", "--port", "8000"]
