# Retraining Orchestrator Dockerfile (Apache Airflow)
FROM apache/airflow:2.8.0-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    postgresql-client \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy DAGs
COPY retraining-orchestrator/dags/ /opt/airflow/dags/
COPY shared/ /opt/airflow/shared/

# Set environment variables
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth

# Initialize Airflow
RUN airflow db init || true

EXPOSE 8080

CMD [" airflow", "webserver"]