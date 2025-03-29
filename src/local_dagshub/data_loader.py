import os
import glob
import tensorflow as tf
import json

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory 
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

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

def load_data():
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    training_data_dir = os.path.join(PROJECT_ROOT, 'data', 'training')
    progress_file = os.path.join(PROJECT_ROOT, 'merge_progress.json')

    os.makedirs(training_data_dir, exist_ok=True)

    current_subset = read_progress(progress_file)
    next_subset = current_subset + 1

    train_files = sorted(glob.glob(os.path.join(raw_data_dir, f'train_subset[1-{next_subset}].tfrecord')))
    valid_files = sorted(glob.glob(os.path.join(raw_data_dir, f'valid_subset[1-{next_subset}].tfrecord')))

    if len(train_files) > current_subset and len(valid_files) > current_subset:
        merge_tfrecords(train_files, os.path.join(training_data_dir, 'train.tfrecord'))
        merge_tfrecords(valid_files, os.path.join(training_data_dir, 'valid.tfrecord'))
        update_progress(progress_file, next_subset)
        print(f"Merged subsets 1 to {next_subset} for both train and valid datasets.")
    else:
        print("No new subsets to add.")

if __name__ == "__main__":
    load_data()
