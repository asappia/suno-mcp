# Suno MCP Server - Docker Image

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        | gpg --dearmor -o /usr/share/keyrings/ngrok.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ngrok.gpg] https://ngrok-agent.s3.amazonaws.com buster main" \
        > /etc/apt/sources.list.d/ngrok.list \
    && apt-get update && apt-get install -y ngrok \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser server.py .
COPY --chown=appuser:appuser suno_client.py .
COPY --chown=appuser:appuser callback_server.py .
COPY --chown=appuser:appuser callback_store.py .
COPY --chown=appuser:appuser suno_response.py .
COPY --chown=appuser:appuser ngrok_tunnel.py .
COPY --chown=appuser:appuser healthcheck.py .
COPY --chown=appuser:appuser docker-entrypoint.sh .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8090

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python healthcheck.py || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
