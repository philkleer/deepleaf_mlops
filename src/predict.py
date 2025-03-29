import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from io import BytesIO
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Sends logs to stdout
)
logger = logging.getLogger(__name__)

# user defined functions and libraries
from src.config import MODEL_PATH, IMG_SIZE
from src.helpers import load_tfrecord_data

# user defined class for f1 score
@tf.keras.utils.register_keras_serializable()  # Register the custom class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def load_trained_model():
    '''
    Loads the trained model from disk with custom object class F1Score
    
    Returns:
    - model: The loaded Keras model.
    '''
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'F1Score': F1Score}
    )

def get_class_labels():
    '''
    Retrieves class labels dynamically from the training data.

    Returns:
    - class_labels (set): Mapping from index to class label.
    '''
    # DISCUSS: Maybe doing this manually is more efficient? Because now we do this everytime, which seems unnecessary
    train_data = load_tfrecord_data('data/training/train.tfrecord')

    # get the data of the loaded train_data
    dataset = train_data[0]

    class_labels = set()

    # decoding one-hot back to class indices
    for images, labels in dataset:
        class_indices = np.argmax(labels, axis=-1)  
        unique_labels = np.unique(class_indices)
        class_labels.update(unique_labels)

    class_labels = sorted(class_labels)
    # cast to plain Python ints
    class_labels = [int(x) for x in class_labels]
    return class_labels

def preprocess_image(file): # nvd06: changed from image_path to file to fit the main.yp and FastAPI setup. 
    '''
    Loads and preprocesses a single image for model prediction.

    Args:
    - file (str): Path to the image.

    Returns:
    - img_array: Preprocessed image array.
    '''
    # local path transfer or via FastAPI
    if isinstance(file, str):  
        # via CLI giving path
        img = image.load_img(file, target_size=IMG_SIZE)
    else: 
        # via FastApi transfer file
        img = image.load_img(BytesIO(file.file.read()), target_size=IMG_SIZE)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_from_path(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single_image(file): 
    '''
    Predicts the class of a single image.

    Args:
    - file (str): Path to the image.

    Returns:
    - Predicted class label.
    '''
    
    # load model
    model = load_trained_model()
    
    # get classes
    class_labels = get_class_labels() 
    
    # preprocessing
    img_array = preprocess_image(file) 
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    return class_labels[class_index]  # Return class label instead of index

if __name__ == '__main__':
    # added to pass argument within terminal
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Run predictions on the test folder by default
    logger.info(f'Running predictions on the test set: {args.image_path}')
    results = predict_single_image(args.image_path)

    logger.info(f'\nPredictions for Test Set: {results}')