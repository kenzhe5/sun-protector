# Образ для Hugging Face Spaces (Docker Space).
# Отличия от корневого Dockerfile: Spaces запускает контейнер от пользователя
# с uid 1000 и ждёт приложение на порту 7860; база знаний (Chroma) не хранится
# в репозитории, поэтому собирается при старте контейнера.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user agent-service/requirements.txt agent-service/requirements.txt
COPY --chown=user mcp-server/requirements.txt mcp-server/requirements.txt
RUN pip install --no-cache-dir -r agent-service/requirements.txt -r mcp-server/requirements.txt

COPY --chown=user agent-service agent-service
COPY --chown=user mcp-server mcp-server
COPY --chown=user frontend frontend

USER user
WORKDIR /app/agent-service
EXPOSE 7860
CMD ["sh", "-c", "python -m app.rag.ingest && exec uvicorn app.main:app --host 0.0.0.0 --port 7860"]
