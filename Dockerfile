
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-runtime.txt .

RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ ./src/
COPY models/ticket_classifier.joblib ./models/ticket_classifier.joblib

RUN mkdir -p /app/data /app/logs \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
