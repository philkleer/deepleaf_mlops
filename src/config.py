import os
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
else:
    print("Warning: .env file not found!")

# Kaggle API Credentials (User-Specific)
GITHUB_ACTOR = os.getenv("GITHUB_ACTOR", "default_user").upper()  # Detect GitHub username

KAGGLE_USERNAME = os.getenv(f"KAGGLE_USERNAME_{GITHUB_ACTOR}")
KAGGLE_KEY = os.getenv(f"KAGGLE_KEY_{GITHUB_ACTOR}") or os.getenv("KAGGLE_KEY", "")

## ADDING: DagsHub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "default_user")
DAGSHUB_KEY = os.getenv("DAGSHUB_KEY", "default_token")

# Define a dataset storage directory inside the user's home
HOME_DIR = os.path.expanduser("~")  
ROOT_DIR = os.path.abspath("./")

# Define paths for training, validation, and testing
TRAIN_PATH = os.path.join(HOME_DIR, ".cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/")
VALID_PATH = os.path.join(HOME_DIR, ".cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/")
TEST_PATH = os.path.join(HOME_DIR, ".cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/test/test")

# Model save path
MODEL_DIR = "./models"  # Directory to store trained models
MODEL_PATH = os.path.join(MODEL_DIR, "production_model.keras")  # Path for saving/loading the model

# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
NUM_CLASSES = 38

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Logging & History paths
LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists

# Generate timestamp for unique history files
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_PATH = os.path.join(LOGS_DIR, f"history_{TIMESTAMP}.json")

# Dagshub credits
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "https://dagshub.com/philkleer/deepleaf_mlops")
MLFLOW_TRACKING_URL = "http://mlflow:5000"
## MLFLOW_TRACKING_URL = 'https://dagshub.com/schytze0/deep_leaf.mlflow'
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME","Plant_Classification_Experiment")
MODEL_DVC='production_model.keras.dvc'
