# Model Registry Service Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared/ /app/shared/

# Copy service code
COPY model-registry/ /app/

# Create directories
RUN mkdir -p /app/models /app/logs

# Expose ports
EXPOSE 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run MLflow server and API
CMD ["sh", "-c", "mlflow server --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB} --default-artifact-root s3://mlflow-artifacts --host 0.0.0.0 --port 5000 & uvicorn main:app --host 0.0.0.0 --port 8000"]