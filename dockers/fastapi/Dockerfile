FROM python:3.8-slim

WORKDIR /app
ENV PYTHONPATH="/app"
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    pkg-config \
    libhdf5-dev \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*
COPY dockers/fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden by Airflow)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]