FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py sandbox_runner.py ./

USER appuser

ENV PORT=8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
