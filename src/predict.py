import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from data_loader import load_data
from config import MODEL_PATH, IMG_SIZE, TEST_PATH

def load_trained_model():
    """
    Loads the trained model from disk.
    
    Returns:
    - model: The loaded Keras model.
    """
    return tf.keras.models.load_model(MODEL_PATH)

def get_class_labels():
    """
    Retrieves class labels dynamically from the training data.

    Returns:
    - class_labels (dict): Mapping from index to class label.
    """
    train_data, _ = load_data()  # Load training data to access class indices
    class_indices = train_data.class_indices  # Dictionary mapping labels to indices
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    return class_labels

def preprocess_image(img_path):
    """
    Loads and preprocesses a single image for model prediction.

    Args:
    - img_path (str): Path to the image.

    Returns:
    - img_array: Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_single_image(img_path):
    """
    Predicts the class of a single image.

    Args:
    - img_path (str): Path to the image.

    Returns:
    - Predicted class label.
    """
    model = load_trained_model()
    class_labels = get_class_labels()  # Get dynamic class labels
    img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    return class_labels[class_index]  # Return class label instead of index

def predict_folder(folder_path):
    """
    Predicts the class for all images in a folder.

    Args:
    - folder_path (str): Path to the folder containing multiple images.

    Returns:
    - results (dict): Dictionary mapping image filenames to predicted labels.
    """
    model = load_trained_model()
    class_labels = get_class_labels()  # Get dynamic class labels
    results = {}

    # Get all image files
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]

    for img_path in image_files:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        results[os.path.basename(img_path)] = class_labels[class_index]

    return results

if __name__ == "__main__":
    # Run predictions on the test folder by default
    print(f"Running predictions on the test set: {TEST_PATH}")
    results = predict_folder(TEST_PATH)

    print("\nPredictions for Test Set:")
    for filename, label in results.items():
        print(f"{filename}: {label}")

