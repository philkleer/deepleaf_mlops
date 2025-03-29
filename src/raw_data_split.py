# INFO: This file is a preliminary step to create 3 train/validation data sets that we might use throughout the program to test different trainings. 
# DEBUG: It works but is not beautiful, however, it is a preliminary step. If we have more time, we probably could spend it on this one. 

import os
import random
import kagglehub
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# User imported
from src.config import IMG_SIZE, BATCH_SIZE, KAGGLE_USERNAME, KAGGLE_KEY, TRAIN_PATH, VALID_PATH

# INFO: Loading original train/validation data set from Kaggle
def setup_kaggle_auth():
    '''
    Sets up Kaggle API authentication dynamically based on `.env` configuration.
    '''
    # Directly use variables from config.py
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY
        print('Kaggle API credentials successfully set.')
    else:
        print('Kaggle API credentials are MISSING. Ensure they are correctly set in the .env file.')

def download_dataset():
    '''
    Downloads the dataset from Kaggle if it's not already available locally.
    '''
    setup_kaggle_auth()  # Ensure Kaggle API authentication is set

    # updated to save data \n the project
    dataset_path = os.path.expanduser('~/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset')

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print('Downloading dataset from KaggleHub...')
        downloaded_path = kagglehub.dataset_download('vipoooool/new-plant-diseases-dataset')
        print(f'Dataset successfully downloaded to: {downloaded_path}')
    else:
        print(f'Dataset already exists at {dataset_path}. Skipping download.')

# Functions to create tfrecord file
def _bytes_feature(value):
    '''
    Returns a bytes_list from a string / byte.
    
    This function converts the input `value` (which can be a string or a byte) 
    into a `tf.train.Feature` object. The value is first encoded into a JPEG format (if it is an image), and then stored as a byte string in a `BytesList` that can be used in a `TFRecord`.

    Arguments:
    - value: A string or byte array. If it's an image, it will be encoded into JPEG format.

    Return:
    - A `tf.train.Feature` object, which contains the byte representation of the input.    
    '''

    # INFO: return back to 255 since `encode_jpeg()` expects this!
    value_uint8 = tf.cast(tf.clip_by_value(value * 255, 0, 255), tf.uint8)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tf.io.encode_jpeg(value_uint8).numpy()]
        )
    )

def _int64_feature(value):
    '''
    Returns an int64_list from a bool / enum / int / uint.

    This function converts the input `value` (which can be a boolean, enum, integer, or unsigned integer) into a `tf.train.Feature` object. The value is stored as an integer in a `Int64List` that can be used in a `TFRecord`.

    Arguments:
    - value: A boolean, enum, integer, or unsigned integer to be stored in the `TFRecord`.

    Return:
    - A `tf.train.Feature` object containing the input value as an `Int64List`.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def create_example(image, label):
    '''
    Creates a tf.train.Example from an image and label.

    This function takes an image and its corresponding label, processes the image into a byte format suitable for storage in a `TFRecord`, and converts the label into an index (for one-hot encoded labels). It then constructs a `tf.train.Example` that contains the image and label as features.

    Arguments:
    - image: A numpy array representing the image to be stored. The image is encoded as a byte string using JPEG encoding.
    - label: A one-hot encoded label (a numpy array) for the image. The label is converted to the scalar index of the '1' in the one-hot encoded vector.

    Return:
    - A `tf.train.Example` object, which encapsulates the image (as bytes) 
      and the label (as an int64) for storage in a `TFRecord`.
    '''

    label_index = int(label)
    
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label_index)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto

def save_as_tfrecord(generator, filename, total_images):

    '''
    Saves data from a generator (train/validation) to a TFRecord file.

    This function processes data from a generator, which yields batches of images and labels, and writes them to a TFRecord file. It uses a progress bar to track the saving process and ensures that the specified number of images is written to the TFRecord file.

    Arguments:
    - generator: A data generator (such as `ImageDataGenerator`) that yields batches of images and labels. Each batch should consist of images and their corresponding labels.
    - filename: The name of the output TFRecord file where the data will be saved.
    - total_images: The total number of images to be processed and saved into the TFRecord file. This value is used for the progress bar and to control the stopping condition of the data writing.

    Return:
    - None: The function writes the images and labels into the specified TFRecord file and does not return anything.
    
    Notes:
    - The function uses `tqdm` to display a progress bar during the saving process.
    - If the generator yields empty batches, they are skipped, ensuring that no empty data is written to the TFRecord.
    - It stops once the specified number of images (`total_images`) is reached.
    '''
    
    # creating the TFrecord files
    with tf.io.TFRecordWriter(filename) as writer:
    # initializing progress bar 
        with tqdm(total=total_images, desc=f'Saving {filename}', unit='image', dynamic_ncols=True) as pbar:
            total_processed = 0

            for images, labels in generator:
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

# Functio to create proper subset for DataGeneraotr
def filtered_generator(subset_data, batch_size=BATCH_SIZE):
    '''
    Custom generator for selected image paths.

    This function generates batches of images and their corresponding labels from a given list of image paths and labels. The images are loaded and preprocessed before being returned in batches. The data is shuffled before batching to ensure randomization.

    Arguments:
    - subset_data: A list of tuples, where each tuple contains the path to an image and its corresponding label. The label can be categorical or numerical.
    - source_dir: The directory where the images are stored. (This argument is included for clarity, but it's not directly used in this function.)
    - batch_size: The number of samples to return in each batch. The default is defined by the constant `BATCH_SIZE`. 

    Return:
    - Yields batches of images and labels in the form of two arrays: one for images and one for labels. Each batch contains `batch_size` samples.

    Notes:
    - The images are resized to a target size defined by `IMG_SIZE`.
    - The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
    - The function ensures that the data is shuffled before batching, providing randomization in each epoch.
    - The generator runs indefinitely, which is typical for training loops in deep learning, and will continue providing batches until manually stopped.

    '''

    images = []
    labels = []

    for img_path in subset_data:
        # image path is tuple of path and label (group)
        img_path, label = img_path
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0 
        images.append(img)
        labels.append(label)

    labels = np.array(labels, dtype=np.int32)
        
    images = np.array(images)

    # Shuffle the data
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices] 

    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size] 
            yield batch_images, batch_labels

# Function to create TFRecord files for subsets
def create_tfrecords(source_dir, tfrecord_paths, split_ratios=None):
    '''
    Splits images into subsets and saves them as TFRecords.

    This function organizes the images from a source directory into subsets, shuffles them, and splits them according to given ratios. Then, it saves the resulting subsets as TFRecord files, which are an efficient format for storing and reading large datasets in TensorFlow.
    For the test data it just creates a TFRecord file.

    Arguments:
    - source_dir: The root directory containing subdirectories of images, where each subdirectory represents a different class. Each class folder should contain images of that class.
    - tfrecord_paths: A dictionary with keys corresponding to subsets (e.g., 'train', 'validation', 'test') and values as the paths where the corresponding TFRecord files should be saved.
    - split_ratios: A tuple representing the proportion of the total dataset to be allocated to each subset. If None, the entire dataset will be used as a single subset. If provided, it can contain any number of values indicating the proportions for each subset.

    Return:
    - Saves the generated TFRecord files to the specified paths in `tfrecord_paths`. The images and their corresponding labels are stored in the TFRecords.
    '''

    if split_ratios is None:
        images = [os.path.join(source_dir, img) for img in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, img))]
        data = [(images), 'all_data']
        
        subset_data = {'all_data': [(img, 'all_data') for img in images]}  
        subset_keys = ['all_data']
                
    else:
        # Checking minimum length if split_ratios is given
        if len(split_ratios) < 1:
            raise ValueError('split_ratios must contain at least one value')

        # checking if sum of split_ratios is 1:
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError('The sum of split_ratios must be 1.')

        # checking if length of split_ratios matches tfrecord_paths
        if len(split_ratios) != len(tfrecord_paths):
            raise ValueError('The length of split_ratios must match the length of tfrecord_paths.')
        
        data = []
        class_labels = sorted(os.listdir(source_dir))
        label_map = {label: idx for idx, label in enumerate(class_labels)}

        subset_data = {key: [] for key in tfrecord_paths.keys()}
        subset_keys = list(tfrecord_paths.keys())
    
        # Collecting image paths and labels from all classes
        for label in class_labels:
            class_dir = os.path.join(source_dir, label)
            if os.path.isdir(class_dir): 
                images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
                int_label = label_map[label]
                data.append((images, int_label))
        
        # Now shuffle the entire dataset (across classes)
        all_images_labels = []
        for class_images, label in data:
            all_images_labels.extend([(img, label) for img in class_images])

        random.shuffle(all_images_labels)  # Shuffle across all classes

        # Split the shuffled data into subsets based on split_ratios
        start = 0
        for i, subset_key in enumerate(subset_keys):
            subset_size = int(len(all_images_labels) * split_ratios[i])
            subset_data[subset_key].extend(all_images_labels[start:start + subset_size])
            start += subset_size

        # Case: remaining files due to rounding would be given to the last subset
        if start < len(all_images_labels):
            remaining_images_labels = all_images_labels[start:]
            subset_data[subset_keys[-1]].extend(remaining_images_labels)
        
    # Saving the TFRecord files
    for subset_key in subset_keys:
        random.shuffle(subset_data[subset_key])
        subset_images = subset_data[subset_key]
        total_images = len(subset_images)
        
        if total_images > 0:
            generator = filtered_generator(subset_images)
            save_as_tfrecord(generator, tfrecord_paths[subset_key], total_images)
        else:
            print(f'Warning: Subset {subset_key} is empty, skipping!')

# DEBUG: I think this approach is not soo good; might be better to have the variables in a pipeline-saving object later on (i.e., in Github Actions)
# TODO: Not yet implemented, I've manually coded the classes in config
# TODO: Think about how to implement this in CI/CD later on
def update_config_num_classes(num_classes, config_file='config.py'):
    ''' 
    Saves the NUM_CLASSES to config.py (for use with tfrecord files)

    Arguments:
    - num_classes [int]: number of classes of train data
    - config_file [str]: path to config.py file
    
    Return:
    - None
    '''
    # check if file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = []

    # check if there is already an entry with NUM_CLASSES (if so updates the line)
    num_classes_exists = False
    for i, line in enumerate(lines):
        if line.startswith('NUM_CLASSES'):
            # case: NUM_CLASSES exists: update line and set num_classes_exists=True
            lines[i] = f'NUM_CLASSES = {num_classes}\n'
            num_classes_exists = True
            break
    
    # case: NUM_CLASSES does not exist
    if not num_classes_exists:
        lines.append(f'NUM_CLASSES = {num_classes}\n')

    # write the new file
    with open(config_file, 'w') as f:
        f.writelines(lines)
    
    print(f'config.py updated with NUM_CLASSES = {num_classes}')

if __name__ == '__main__':
    print('Beginning data file download ...', end='\r')
    download_dataset()
    print('Data file downloaded/already present ✅')
    
    # Defining output directories
    output_dir = 'data/raw/'
    os.makedirs(output_dir, exist_ok=True)

    subsets = ['subset1', 'subset2', 'subset3', 'subset4', 'subset5', 'subset6', 'subset7', 'subset8', 'subset9', 'subset10']
    
    tfrecord_paths = {
        'train': {subset: os.path.join(output_dir, f'train_{subset}.tfrecord') for subset in subsets},
        'valid': {subset: os.path.join(output_dir, f'valid_{subset}.tfrecord') for subset in subsets},
        'test':  {'all_data': os.path.join(output_dir, 'test.tfrecord')}
    }

    split_ratios = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    
    # Training data: splitting, transforming (tfrecord) and saving
    print('Beginning subsetting training data ...', end='\r')
    create_tfrecords(TRAIN_PATH, tfrecord_paths['train'], split_ratios)
    print(f'Training data is subsetted and saved to {output_dir} ✅')

    # Validation data: splitting, transforming (tfrecord) and saving
    print('Beginning subsetting validation data ...', end='\r')
    create_tfrecords(VALID_PATH, tfrecord_paths['valid'], split_ratios)
    print(f'Validation data is subsetted and saved to {output_dir} ✅')
