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

# Define a dataset storage directory inside the user's home
HOME_DIR = os.path.expanduser("~")  
# ROOT_FOLDER = os.path.join(HOME_DIR, ".cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2")
# DATA_DIR = os.path.join(ROOT_FOLDER, "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/")
ROOT_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(ROOT_DIR, "data/")

# Define paths for training, validation, and testing
TRAIN_PATH = os.path.join(DATA_DIR, "raw/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/")
VALID_PATH = os.path.join(DATA_DIR, "raw/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/")
TEST_PATH = os.path.join(ROOT_DIR, "raw/versions/2/test/test/")
# added processed path
PROC_DIR = os.path.join(DATA_DIR, "processed/")

# Model save path
MODEL_DIR = "./models"  # Directory to store trained models
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")  # Path for saving/loading the model

# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Logging & History paths
LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists

# Generate timestamp for unique history files
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_PATH = os.path.join(LOGS_DIR, f"history_{TIMESTAMP}.json")
NUM_CLASSES = 38
