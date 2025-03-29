import os
import mlflow
import mlflow.keras
import shutil
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import tenacity
import logging

# local imports
from src.config import MODEL_DVC, DAGSHUB_REPO
from src.model_and_data_tracking import track_model_and_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Sends logs to stdout
)
logger = logging.getLogger(__name__)

def load_environment():
    """
    Loads MLflow settings from .env file
    
    Returns:
    - mlflow_url: URL of mlflow container 
    - experiment_name: experiment name
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, "..", ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        logger.info('.env file found and loaded ✅')
    else:
        raise FileNotFoundError("⛔️ Warning: .env file not found!")
    
    mlflow_url = os.getenv('MLFLOW_TRACKING_URI')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
    
    if not mlflow_url or not experiment_name:
        raise ValueError(
            "MLFLOW_TRACKING_URI or MLFLOW_EXPERIMENT_NAME not set in .env"
        )
    
    logger.info(f'Loaded credentials:\n- URI: {mlflow_url}\n- Experiment name: {experiment_name}')
    
    return mlflow_url, experiment_name

def get_best_model(mlflow_url, experiment_name):
    """
    Retrieves the best model from MLFlow based on validation accuracy.

    Arguments:
        - mlflow_uri: loaded MLflow URI from environment (str)
        - experiment_name: loaded experiment name from environment (str)
    Returns:
        tuple: (model_path, best_val_accuracy, run_id)
    """

    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(experiment_name)
    
    best_run = mlflow.search_runs(
        order_by=["metrics.final_val_accuracy DESC"]
    ).head(1)
    if best_run.empty:
        logger.info("No runs found in the experiment")
        return None, None, None
    
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['metrics.final_val_accuracy']
    
    logger.info(f"Run ID: {run_id}, Validation accuracy: {best_val_accuracy:.4f}")

    model_uri = f"runs:/{run_id}/model"
    
    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )
    logger.info(f"Model downloaded to: {local_model_path}")
    # Debug: List files to confirm structure
    artifact_files = list(Path(local_model_path).rglob("*"))
    logger.info(f"Artifact files: {artifact_files}")
    return local_model_path, best_val_accuracy, run_id

def check_metadata_exists():
    """Check if metadata.txt exists in repo root and get its accuracy."""
    repo_root = Path.cwd()

    metadata_path = repo_root / "models/metadata.txt"

    dvc_pointer_path = repo_root / f'models/{MODEL_DVC}'
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            try:
                accuracy = float(f.read().strip())
                logger.info(f"Existing metadata found with accuracy: {accuracy:.4f}")
                return True, accuracy
            except ValueError as e:
                logger.info(f"Malformed metadata.txt: {e}. Assuming accuracy 0.0")
                return True, 0.0
    elif dvc_pointer_path.exists():
        logger.info(f"Found {MODEL_DVC} but no metadata.txt; assuming accuracy 0.0")
        return True, 0.0
    else:
        logger.info(f"No metadata.txt or {MODEL_DVC} found; assuming initial accuracy of 0.0")
        return False, 0.0

def overwrite_existing(model_artifact_path, val_accuracy):
    """
    Copy the best model file into /models/, create a DVC pointer file (models.dvc) and metadata.txt in the repo root.
    """
    repo_root = Path.cwd()
    
    target_dir = repo_root / "models"  
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Directly fetch model.keras from the data/ subdirectory
    source_model_file = repo_root / "temp" / "current_model.keras"
    if not source_model_file.exists():
        artifact_files = list(Path("app/temp/").rglob("*"))
        logger.error(f"Error: No model.keras found at {source_model_file}. Available files: {artifact_files}")
        raise FileNotFoundError(f"No model.keras found in {model_artifact_path}/data")
    
    dest_model_file = repo_root / "models" / "production_model.keras"
    shutil.copy2(source_model_file, dest_model_file)
    logger.info(f"Copied model to {dest_model_file}")

     # Update metadata.txt in models (file is component of production model)
    metadata_path = repo_root / "models" / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(str(val_accuracy))
    logger.info(f"Updated {metadata_path} with accuracy: {val_accuracy:.4f}")
    

def update_model_if_better():
    try:
        mlflow_url, experiment_name = load_environment()
    except ValueError as e:
        return str(e)

    # Get the path and the val_accuracy of the best model of the finished training    
    best_model_path, best_val_accuracy, run_id = get_best_model(mlflow_url, experiment_name)
    
    if not best_model_path:
        return "No models found in MLFlow experiments"
    
    exists, existing_accuracy = check_metadata_exists()
    action = "No change"
    
    if not exists or best_val_accuracy > existing_accuracy:
        action = "Created" if not exists else "Updated"
        try:
            overwrite_existing(best_model_path, best_val_accuracy)
            track_model_and_data()
            return f"Model {action.lower()} - New accuracy: {best_val_accuracy:.4f}" + (f", Previous: {existing_accuracy:.4f}" if exists else "")
        except Exception as e:
            logger.info(f"Error during upload: {e}")
            return f"Failed to {action.lower()} model; see manual instructions"
    else: 
        return f"No change - Existing accuracy ({existing_accuracy:.4f}) >= Current ({best_val_accuracy:.4f})"

if __name__ == "__main__":
    result = update_model_if_better()
    logger.info(f"Model management result: {result}")