import os
import tensorflow as tf

# Import your custom F1Score class.
try:
    from src.predict import F1Score
except ImportError as e:
    print("Error importing F1Score:", e)
    raise

# Set the model path. Adjust if needed.
MODEL_PATH = "./models/production_model.keras"

print("TensorFlow version:", tf.__version__)
print("Current working directory:", os.getcwd())
print("Looking for model file at:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("Model file not found at", MODEL_PATH)
else:
    print("Model file found. Attempting to load the model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'F1Score': F1Score})
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
