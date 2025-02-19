import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_PATH, VALID_PATH, IMG_SIZE, BATCH_SIZE, KAGGLE_USERNAME, KAGGLE_KEY, PROC_DIR
import shutil
import numpy as np
# adding progress bar
from tqdm import tqdm  # Import tqdm for progress bar


def setup_kaggle_auth():
    """
    Sets up Kaggle API authentication dynamically based on `.env` configuration.
    """
    # Directly use variables from config.py
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
        print("Kaggle API credentials successfully set.")
    else:
        print("Kaggle API credentials are MISSING. Ensure they are correctly set in the .env file.")

def download_dataset():
    """
    Downloads the dataset from Kaggle if it's not already available locally.
    """
    setup_kaggle_auth()  # Ensure Kaggle API authentication is set

    # updated to save data in the project
    dataset_path = os.path.expanduser("./data/raw")
    download_temp_path = os.path.expanduser("~/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset")

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("Downloading dataset from KaggleHub...")
        downloaded_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

        print(f"Dataset successfully downloaded to: {downloaded_path}")
        shutil.move(download_temp_path, dataset_path)
        print(f"Raw data successfully transfered to the project at: { dataset_path}")
    else:
        print("Dataset already exists at {dataset_path}. Skipping download.")

# Additional functions to save train and validation data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tf.io.encode_jpeg(value).numpy()]
        )
    )

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_example(image, label):
    """Creates a tf.train.Example from an image and label."""
    # Convert one-hot encoded label to a scalar index
    label_index = np.argmax(label)  # Extract the index of the '1' in the one-hot encoded vector
    
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label_index)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto

def save_as_tfrecord(generator, filename):
    """Saves data from a generator (train/validation) to a TFRecord file."""

    # get total length for status bar
    total_images = generator.samples
    
    # creating the TFrecord files
    with tf.io.TFRecordWriter(filename) as writer:
    # initializing progress bar 
        with tqdm(total=total_images, desc="Saving TFRecord", unit="image", dynamic_ncols=True) as pbar:
            total_processed = 0

            for batch_idx, (images, labels) in enumerate(generator):
                # if empty go to next
                if len(images) == 0 or len(labels) == 0:
                    continue  

                # processing each image-label pair
                for image, label in zip(images, labels):
                    example = create_example(image, label)
                    writer.write(example.SerializeToString())
                    # adding increment
                    total_processed += 1  

                    # updating progress bar
                    pbar.update(1)

                # check if we have processed all images
                # error check: if this is not included it goes above the number of images included, which I don't understand
                if total_processed >= total_images:
                    break


def update_config_num_classes(num_classes, config_file='config.py'):
    """ 
    Saves the NUM_CLASSES to config.py (for use with tfrecord files)

    Arguments:
    - num_classes [int]: number of classes of train data
    - config_file [str]: path to config.py file
    
    Return:
    - None
    """
    # check if file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = []

    # check if there is already an entry with NUM_CLASSES (if so updates the line)
    num_classes_exists = False
    for i, line in enumerate(lines):
        if line.startswith("NUM_CLASSES"):
            # case: NUM_CLASSES exists: update line and set num_classes_exists=True
            lines[i] = f"NUM_CLASSES = {num_classes}\n"
            num_classes_exists = True
            break
    
    # case: NUM_CLASSES does not exist
    if not num_classes_exists:
        lines.append(f"NUM_CLASSES = {num_classes}\n")

    # write the new file
    with open(config_file, 'w') as f:
        f.writelines(lines)
    
    print(f"config.py updated with NUM_CLASSES = {num_classes}")

def load_data():
    """
    Loads dataset using separate paths for training and validation.
    """
    download_dataset()  # Ensure the dataset is available

    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        VALID_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    # save number of classe
    num_classes = train_data.num_classes
    # writing num classes to config.py
    update_config_num_classes(num_classes, config_file='src/config.py')

    # saving the generated data
    os.makedirs(PROC_DIR, exist_ok=True)

    save_as_tfrecord(train_data, os.path.join(PROC_DIR, "train_data.tfrecord"))
    save_as_tfrecord(val_data, os.path.join(PROC_DIR, "val_data.tfrecord"))

    return train_data, val_data

if __name__ == "__main__":
    # Test loading data
    train_data, val_data = load_data()
    print(f"Train samples: {train_data.samples}, Validation samples: {val_data.samples}")

