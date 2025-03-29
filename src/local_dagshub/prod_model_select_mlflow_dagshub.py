import os
import mlflow
import mlflow.keras
import shutil
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import tenacity

# imports from config
from src.config import MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME, DAGSHUB_REPO, MODEL_DVC

# Ensure script runs from this file path (or any of yours)
# INFO: CLONE_DIR must be saved in your .env file
os.chdir(os.getenv("CLONE_DIR")) 

def load_environment():
    """
    Load environment variables from .env file
    
    Returns:
    - username: DAGSHUB_USERNAME from .env
    - token: DAGSHUB_TOKEN from .env
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, "..", ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        print('.env file found and loaded ✅')
    else:
        raise FileNotFoundError("Warning: .env file not found!")
    
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_KEY')
    
    if not username or not token:
        raise ValueError("DAGSHUB_USERNAME or DAGSHUB_KEY not set in .env")
    
    print(f"Loaded credentials - Username: {username}")
    return username, token

def get_best_model():
    """
    Retrieves the best model from MLFlow based on validation accuracy.
    
    Returns:
        tuple: (model_path, best_val_accuracy, run_id)
    """
    # INFO: MLFLOW_TRACKING_URL and MLFLOW_EXPERIMENT_NAME are saved in config.py
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # INFO: Since we have two runs (untrained/fine-tuned) we take the last validation accuracy from the fine-tuned model.
    best_run = mlflow.search_runs(order_by=["metrics.final_val_accuracy DESC"]).head(1)
    if best_run.empty:
        print("No runs found in the experiment")
        return None, None, None
    
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['metrics.final_val_accuracy']
    
    print(f"Run ID: {run_id}, Validation accuracy: {best_val_accuracy:.4f}")

    model_uri = f"runs:/{run_id}/model"
    
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    print(f"Model downloaded to: {local_model_path}")
    # Debug: List files to confirm structure
    artifact_files = list(Path(local_model_path).rglob("*"))
    print(f"Artifact files: {artifact_files}")
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
                print(f"Existing metadata found with accuracy: {accuracy:.4f}")
                return True, accuracy
            except ValueError as e:
                print(f"Malformed metadata.txt: {e}. Assuming accuracy 0.0")
                return True, 0.0
    elif dvc_pointer_path.exists():
        print(f"Found {MODEL_DVC} but no metadata.txt; assuming accuracy 0.0")
        return True, 0.0
    else:
        print(f"No metadata.txt or {MODEL_DVC} found; assuming initial accuracy of 0.0")
        return False, 0.0

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def dvc_push(repo_root):
    """Retry DVC push up to 3 times with exponential backoff."""
    subprocess.run(["dvc", "push"], cwd=repo_root, check=True)
    print("Pushed model to DVC remote ✅")

    # need of git push after dvc push (change of dvc files)
    subprocess.run(['git', 'push'], cwd=repo_root, check=True)
    print("Pushed dvc changes to git ✅")

def add_or_modify_remote_with_auth(dvc_remote, repo_root, dagshub_repo, token):
    '''
    Checks if remote is already there or not. 

    Inputs:
        - dvc_remote: branch in dagshub
        - repo_root: root path
        - dagshub_repo: url of repo
    
    Returns: none
    '''
    try:
        # Check if the remote already exists
        result = subprocess.run(
            ['dvc', 'remote', 'list', repo_root],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        ).stdout

        # Check if correct remote and url exist
        lines = result.splitlines()
        remote_exists = False
        correct_url = False

        for line in lines:
            name, url = line.split(maxpslit=1)
            if name == dvc_remote:
                remote_exists = True
                if url == 's3://dvc':
                    correct_url = True

        # case remote and correct url exist
        if remote_exists and correct_url:
            # modify endpoint
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, 'endpointurl', dagshub_repo],
                cwd=repo_root,
                check=True
            )

            # modify authentification
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'access_key_id', token], 
                cwd=repo_root, 
                check=True
            )
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'secret_access_key', token], 
                cwd=repo_root, 
                check=True)

            print(f'Remote {dvc_remote} modified with authentification successfully.')
        elif remote_exists and not correct_url:
            # modify endpoint
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, 'url', 's3://dvc'],
                cwd=repo_root,
                check=True
            )

            # modify endpoint
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, 'endpointurl', dagshub_repo],
                cwd=repo_root,
                check=True
            )

            # modify authentification
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'access_key_id', token], 
                cwd=repo_root, 
                check=True
            )
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'secret_access_key', token], 
                cwd=repo_root, 
                check=True)

            print(f'Remote {dvc_remote} modified (endpointurl) with authentification successfully.')
        else:
            # Case neither remote nor url exist or url exist but under wrong remote name
            # modify endpoint
            subprocess.run(
                ['dvc', 'remote', 'add', dvc_remote, 'url', 's3://dvc'],
                cwd=repo_root,
                check=True
            )

            # modify endpoint
            subprocess.run(
                ['dvc', 'remote', 'add', dvc_remote, 'endpointurl', dagshub_repo],
                cwd=repo_root,
                check=True
            )

            # modify authentification
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'access_key_id', token], 
                cwd=repo_root, 
                check=True
            )
            subprocess.run(
                ['dvc', 'remote', 'modify', dvc_remote, '--local', 'secret_access_key', token], 
                cwd=repo_root, 
                check=True)

            print(f'Remote {dvc_remote} added with authentification successfully.')

    except subprocess.CalledProcessError as e:
        print(f'Error during remote configuration: {e}')

def upload_model_to_dagshub(username, token, model_artifact_path, val_accuracy, branch="main"): 
    """
    Copy the best model file into src/main/models/, create a DVC pointer file (models.dvc) and metadata.txt in the repo root, then commit and push changes.
    """
    repo_root = Path.cwd()
    
    target_dir = repo_root / "models"  
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Directly fetch model.keras from the data/ subdirectory
    source_model_file = Path(model_artifact_path) / "data" / "model.keras"
    if not source_model_file.exists():
        artifact_files = list(Path(model_artifact_path).rglob("*"))
        print(f"Error: No model.keras found at {source_model_file}. Available files: {artifact_files}")
        raise FileNotFoundError(f"No model.keras found in {model_artifact_path}/data")
    
    dest_model_file = target_dir / "production_model.keras"
    shutil.copy2(source_model_file, dest_model_file)
    print(f"Copied model to {dest_model_file}")
    
    # Track with DVC
    subprocess.run(["dvc", "add", str(dest_model_file)], cwd=repo_root, check=True)
    dvc_file = target_dir / MODEL_DVC
    target_dvc_file = repo_root / MODEL_DVC
    shutil.move(dvc_file, target_dvc_file)
    print(f"Moved DVC pointer to {target_dvc_file}")
    
    # Update metadata.txt in models (file is component of production model)
    metadata_path = repo_root / "models/metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(str(val_accuracy))
    print(f"Updated {metadata_path} with accuracy: {val_accuracy:.4f}")
    
    # Git commit and push
    subprocess.run(["git", "add", str(target_dvc_file), str(metadata_path)], cwd=repo_root, check=True)
    commit_msg = f"Update model with accuracy {val_accuracy:.4f}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=repo_root, check=True)
    print("Pushed Git changes")
    
    # Configure DVC remote and push with retry
    dvc_remote = "origin"
    TOKEN = os.getenv('DAGSHUB_KEY')

    add_or_modify_remote_with_auth(dvc_remote, repo_root, DAGSHUB_REPO, TOKEN)
    
    try:
        dvc_push(repo_root)
    except Exception as e:
        print(f"❌ DVC push failed after retries: {e}")
        raise
    
    return True

def manual_upload_instructions(username, model_path, val_accuracy):
    print("\n=== MANUAL UPLOAD INSTRUCTIONS ===")
    print(f"1. Locate model.keras in: {model_path}/data")
    print("2. Rename to 'plant_disease_model.keras' and move to 'models/'")  # If main branch uses src/models/, update to 'src/models/'.
    print("3. Run: dvc add models/production_model.keras")  # If main branch uses src/models/, update to 'dvc add src/models/plant_disease_model.keras'.
    print(f"4. Create/update models/metadata.txt with: {val_accuracy}")
    print("5. Run: git add models.dvc models/metadata.txt")
    print("6. Run: git commit -m 'Update model' && git push origin main")  # Change "dev-erwin" to "main" after merging into main branch.
    print("7. Run: dvc push")
    print("===============================\n")

def update_model_if_better():
    username, token = load_environment()
    best_model_path, best_val_accuracy, run_id = get_best_model()
    
    if not best_model_path:
        return "No models found in MLFlow experiments"
    
    exists, existing_accuracy = check_metadata_exists()
    action = "No change"
    
    if not exists or best_val_accuracy > existing_accuracy:
        action = "Created" if not exists else "Updated"
        try:
            upload_model_to_dagshub(username, token, best_model_path, best_val_accuracy)
            return f"Model {action.lower()} - New accuracy: {best_val_accuracy:.4f}" + (f", Previous: {existing_accuracy:.4f}" if exists else "")
        except Exception as e:
            print(f"Error during upload: {e}")
            manual_upload_instructions(username, best_model_path, best_val_accuracy)
            return f"Failed to {action.lower()} model; see manual instructions"
    return f"No change - Existing accuracy ({existing_accuracy:.4f}) >= Current ({best_val_accuracy:.4f})"

if __name__ == "__main__":
    result = update_model_if_better()
    print(f"Model management result: {result}")