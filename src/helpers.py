import tensorflow as tf
from config import NUM_CLASSES, BATCH_SIZE

# INFO: Just another file to store functions that might be used in more than a single script
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

    # One-hot encode the label (assuming 38 classes)
    label = tf.one_hot(parsed_features['label'], depth=NUM_CLASSES)

    return image, label

def load_tfrecord_data(tfrecord_file):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Map the parsing function over the dataset
    dataset = dataset.map(_parse_function)

    # Shuffle, batch, and prefetch the dataset for performance
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
