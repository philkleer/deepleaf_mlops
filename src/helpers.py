import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# User imported
from src.config import NUM_CLASSES, BATCH_SIZE, IMG_SIZE

# including saved train_/val_data.tfrecord
def _parse_function(proto):
    # Define the feature structure
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input `tf.train.Example` proto using the feature structure
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Decode the image data
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)

    # Resize to (224, 224)
    image = tf.image.resize(image, IMG_SIZE)  
    
    # cast image and rescale
    image = tf.cast(image, tf.float32)

    # Preprocess the image using VGG16-specific preprocessing
    image = preprocess_input(image)

    # ensuring labels are in range [0, NUM_CLASSES - 1]
    label = tf.clip_by_value(parsed_features['label'], 0, NUM_CLASSES - 1)
    
    # One-hot encode the label (assuming 38 classes)
    label = tf.one_hot(parsed_features['label'], depth=NUM_CLASSES)

    return image, label

def load_tfrecord_data(tfrecord_file):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Map the parsing function over the dataset
    dataset = dataset.map(_parse_function)

    # Shuffle, batch, and prefetch the dataset for performance
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    total_records = sum(1 for _ in dataset.unbatch())

    return dataset, total_records
