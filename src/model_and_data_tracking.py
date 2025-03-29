import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def configure_git_identity():
    user_email = os.getenv("GIT_USER_EMAIL")
    user_name = os.getenv("GIT_USER_NAME")
    if not user_email or not user_name:
        raise EnvironmentError("GIT_USER_EMAIL and/or GIT_USER_NAME not set in environment variables.")
    subprocess.run(f"git config --global user.email '{user_email}'", shell=True, check=True)
    subprocess.run(f"git config --global user.name '{user_name}'", shell=True, check=True)
    logger.info(f"Configured Git identity: {user_name} <{user_email}>")

def get_current_branch():
    result = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True, text=True, check=True)
    branch = result.stdout.strip()
    logger.info(f"Current Git branch detected: {branch}")
    return branch

def commit_changes():
    result = subprocess.run("git diff --cached --quiet", shell=True)
    if result.returncode != 0:
        subprocess.run("git commit -m 'AUTOMATED: Update production model with improved val_accuracy:'", shell=True, check=True)
        logger.info("Git commit successful.")
    else:
        logger.info("No changes detected to commit.")

def track_model_and_data():
    try:
        # Configure Git identity
        configure_git_identity()

        # Update DVC tracking for the production model
        subprocess.run("dvc add models/production_model.keras", shell=True, check=True)
        logger.info("DVC tracking updated for production_model.keras.")

        # Update DVC tracking for the data folder
        subprocess.run("dvc add data", shell=True, check=True)
        logger.info("DVC tracking updated for data folder.")

        current_branch = get_current_branch()

        # Stage the DVC pointer files and metadata
        subprocess.run("git add models/production_model.keras.dvc data.dvc models/metadata.txt", shell=True, check=True)
        commit_changes()
        subprocess.run(f"git push origin {current_branch}", shell=True, check=True)
        logger.info("Git changes pushed successfully.")

        # Push the model file to the remote
        subprocess.run("dvc push", shell=True, check=True)
        logger.info("DVC push completed successfully.")

        # Checkout the model file so it is actually available locally
        subprocess.run("dvc checkout models/production_model.keras data", shell=True, check=True)
        logger.info("DVC checkout completed for production_model.keras and data folder.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in track_model: {e}")
        raise

if __name__ == '__main__':
    track_model_and_data()
