import os
import glob
import tensorflow as tf
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Sends logs to stdout
)
logger = logging.getLogger(__name__)

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def merge_tfrecords(input_files, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for input_file in input_files:
            for record in tf.data.TFRecordDataset(input_file):
                writer.write(record.numpy())

def update_progress(progress_file, current_subset):
    with open(progress_file, 'w') as f:
        json.dump({'current_subset': current_subset}, f)

def read_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)['current_subset']
    return 0

def create_data():
    raw_data_dir = os.path.join('data', 'raw')
    training_data_dir = os.path.join('data', 'training')
    progress_file = os.path.join('merge_progress.json')

    os.makedirs(training_data_dir, exist_ok=True)

    current_subset = read_progress(progress_file)
    next_subset = current_subset + 1

    train_files = sorted(glob.glob(os.path.join(raw_data_dir, f'train_subset[1-{next_subset}].tfrecord')))
    valid_files = sorted(glob.glob(os.path.join(raw_data_dir, f'valid_subset[1-{next_subset}].tfrecord')))

    if len(train_files) > current_subset and len(valid_files) > current_subset:
        merge_tfrecords(train_files, os.path.join(training_data_dir, 'train.tfrecord'))
        merge_tfrecords(valid_files, os.path.join(training_data_dir, 'valid.tfrecord'))
        update_progress(progress_file, next_subset)
        logger.info(f"Merged subsets 1 to {next_subset} for both train and valid datasets. ✅")
    else:
        logger.info("No new subsets to add. ✅")

if __name__ == "__main__":
    create_data()
