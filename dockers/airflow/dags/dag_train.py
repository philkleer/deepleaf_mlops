from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os
from pathlib import Path
from docker.types import Mount

host_repo_path = os.getenv("REPO_ROOT")
if not host_repo_path:
    raise ValueError("REPO_ROOT environment variable not set!")

ssh_folder_path = os.path.join(host_repo_path, "ssh")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 27),  # Updated to today for testing
    'end_date': datetime(2025, 4, 5),   # 10 days from start_date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'model_training_and_update',
    tags=['airflow', 'deep_leaf'],
    default_args=default_args,
    description='Train model in fastapi-app container',
    schedule_interval='0 22 * * *',  # for daily at 10pm use: '0 22 * * *'
    catchup=False,
    max_active_runs=1,
)

# Task to train the model
train_model = DockerOperator(
    task_id='train_model',
    image='fastapi-app:latest',
    command='python /app/src/train.py',
    docker_url='unix://var/run/docker.sock',
    auto_remove=True,
    network_mode='bridge',
    working_dir='/app',
    do_xcom_push=True,
    mounts=[Mount(source=host_repo_path, target='/app', type='bind'),
            Mount(source=ssh_folder_path, target="/root/.ssh", type="bind")
            ],
    mount_tmp_dir=False,
    dag=dag,
)
