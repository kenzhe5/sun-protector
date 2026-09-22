FROM python:3.11-slim
WORKDIR /app

COPY agent-service/requirements.txt agent-service/requirements.txt
COPY mcp-server/requirements.txt mcp-server/requirements.txt
RUN pip install --no-cache-dir -r agent-service/requirements.txt -r mcp-server/requirements.txt

COPY agent-service agent-service
COPY mcp-server mcp-server
COPY frontend frontend

WORKDIR /app/agent-service
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
