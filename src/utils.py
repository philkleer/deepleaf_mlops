import json
import os
import matplotlib.pyplot as plt

# User imported
from src.config import LOGS_DIR

def get_latest_history_file():
    """
    Finds the most recent history file in the logs directory.

    Returns:
    - history_file (str): Path to the latest history file.
    """
    history_files = [f for f in os.listdir(LOGS_DIR) if f.startswith("history_") and f.endswith(".json")]
    if not history_files:
        print("No training history found. Train a model first.")
        return None

    # Sort history files by timestamp (most recent first)
    history_files.sort(reverse=True)
    return os.path.join(LOGS_DIR, history_files[0])

def load_training_history():
    """
    Loads training history from the latest JSON file.

    Returns:
    - history (dict): Dictionary containing training accuracy/loss.
    """
    history_file = get_latest_history_file()
    if history_file is None:
        return None

    with open(history_file, "r") as f:
        history = json.load(f)

    return history

def plot_training_history():
    """
    Plots training accuracy and loss over epochs.
    """
    history = load_training_history()
    if history is None:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax[0].plot(history['accuracy'], label='Train Accuracy')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss plot
    ax[1].plot(history['loss'], label='Train Loss')
    ax[1].plot(history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

if __name__ == "__main__":
    plot_training_history()

