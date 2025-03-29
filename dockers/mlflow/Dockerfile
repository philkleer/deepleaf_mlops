FROM python:3.8-slim

RUN pip install mlflow psycopg2-binary
RUN apt-get update && apt-get install -y  gcc python3-dev libpq-dev
RUN pip install dvc dvc-s3

EXPOSE 5001

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow-postgres/mlflow", "--default-artifact-root", "/app/mlflow/artifacts"]