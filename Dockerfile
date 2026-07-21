# Serving image for the FastAPI scoring endpoint.
# Training deps (tensorflow etc.) are included so the same image can also run
# `python -m src.train`; slim variants can strip these if desired.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for psycopg2 + scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# The model artifacts are expected under /app/models (mount or bake them in).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
