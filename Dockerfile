# Not tested in the build sandbox. Run `python ingest.py` locally first so ./storage exists.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Reach Ollama running on your host machine
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# docker build -t support-copilot .
# docker run -p 8000:8000 -v "$(pwd)/storage:/app/storage" --add-host=host.docker.internal:host-gateway support-copilot
