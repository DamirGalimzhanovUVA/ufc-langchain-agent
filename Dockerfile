FROM ghcr.io/ggml-org/llama.cpp:server@sha256:4f02c560799a1569be08b0183d52b94b0d4a6e4b88f52f20562d2334c73837d4 AS llama

FROM node:24-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="/opt/llama" \
    PATH="/opt/venv/bin:/opt/llama:$PATH"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgomp1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=llama /app/ /opt/llama/

COPY backend/requirements.txt backend/requirements.txt
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --requirement backend/requirements.txt

COPY frontend/package.json frontend/package-lock.json frontend/
RUN npm ci --prefix frontend

COPY . .
RUN chmod +x launch.sh backend/start-llama-server.sh

EXPOSE 5173 8000

CMD ["./launch.sh"]
