import os
from dotenv import load_dotenv
from config import (
    KAGGLE_USERNAME, KAGGLE_KEY, TRAIN_PATH, VALID_PATH, TEST_PATH,
    MODEL_PATH, LOGS_DIR, HISTORY_PATH, GITHUB_ACTOR
)
# from data_loader import download_dataset
from raw_data_split import download_dataset

# Force loading .env BEFORE importing config
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

print("Checking Configuration Settings...\n")

# Test Kaggle API Credentials
print(f"GitHub Actor: {GITHUB_ACTOR}")
if KAGGLE_USERNAME and KAGGLE_KEY:
    print("Kaggle API credentials are set.")
else:
    print("Kaggle API credentials are MISSING. Check GitHub Secrets.")

# Test Dataset Paths
print(f"\nTrain Path: {TRAIN_PATH}")
print(f"Valid Path: {VALID_PATH}")
print(f"Test Path: {TEST_PATH}")

if os.path.exists(TRAIN_PATH) and os.path.exists(VALID_PATH):
    print("Training and validation directories exist.")
else:
    print("Training data directory NOT found.")

    # Automatically Download Dataset if Missing
    user_input = input("Do you want to download the dataset from Kaggle now? (yes/no): ").strip().lower()
    if user_input in ["yes", "y"]:
        print("Downloading dataset from Kaggle...")
        download_dataset()  # Call the dataset download function
        print("Dataset downloaded successfully.")
    else:
        print("Dataset was NOT downloaded. Run `python data_loader.py` manually to download it.")

if os.path.exists(TRAIN_PATH):
    print("Training data directory exists.")
else:
    print("Training data directory NOT found.")

# Test Model Path
print(f"\nModel Path: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print("Model file exists.")
else:
    print("Model file NOT found (expected for first-time training).")

# Test Logs Directory
print(f"\nLogs Directory: {LOGS_DIR}")
if os.path.exists(LOGS_DIR):
    print("Logs directory exists.")
else:
    print("Logs directory does NOT exist. Creating it now...")
    os.makedirs(LOGS_DIR)

# Test History File Path
print(f"\nTraining History Path: {HISTORY_PATH}")
print("\nConfiguration check complete.")

