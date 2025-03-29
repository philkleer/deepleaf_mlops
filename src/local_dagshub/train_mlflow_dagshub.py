import tensorflow as tf
from tensorflow.keras import optimizers, callbacks  
import json
import os
import mlflow
import mlflow.keras
from dotenv import load_dotenv
from pydantic import BaseModel
import subprocess 
from pathlib import Path

# imports from other scripts
from src.config import HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES, MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME
from src.model import build_vgg16_model
from src.helpers import load_tfrecord_data
from src.prod_model_select import update_model_if_better

# new class F1-Score
# F1 score to get better reporting
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

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Access dagshub 
# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, "..", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
    print('.env file found and loaded ✅')
else:
    print("Warning: .env file not found!")

# Debugging: Print environment variables to verify they're loaded
dagshub_username = os.getenv('DAGSHUB_USERNAME')
dagshub_key = os.getenv('DAGSHUB_KEY')

if not dagshub_username:
    print("❌ ERROR: DAGSHUB_USERNAME is not set.")
if not dagshub_key:
    print("❌ ERROR: DAGSHUB_KEY is not set.")

# ML Flow setup (still needs to be tested)
class MLFlowLogger(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0
        self.final_val_accuracy = 0
        self.final_val_f1_score = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric('train_loss', logs.get('loss'), step=epoch)
            mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=epoch)
            mlflow.log_metric('train_f1_score', logs.get('f1_score'), step=epoch)
            mlflow.log_metric('val_loss', logs.get('val_loss'), step=epoch)
            mlflow.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch)
            mlflow.log_metric('val_f1_score', logs.get('val_f1_score'), step=epoch)

            # checking if it is the best epoch based on validation
            val_accuracy = logs.get('val_accuracy')
            val_f1_score = logs.get('val_f1_score')

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_f1_score = val_f1_score
                self.best_epoch = epoch
                mlflow.log_metric('best_val_accuracy', self.best_val_accuracy)
                mlflow.log_metric('best_val_f1_score', self.best_val_f1_score)
                mlflow.log_metric('best_epoch', self.best_epoch)
                print(f'Updated best validation accuracy: {round(val_accuracy, 4)} ✅')

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}

        # Logging of final scores (depending on what to do we might be interested in best value throughout epochs or the final score)
        self.final_val_accuracy = logs.get("val_accuracy")
        self.final_val_f1_score = logs.get("val_f1_score")

        mlflow.log_metric('final_val_accuracy', self.final_val_accuracy)
        mlflow.log_metric('final_val_f1_score', self.final_val_f1_score)

        # print best values
        print(f'Best Validation Accuracy: {self.best_val_accuracy:.4f}')
        print(f'Final validation accuracy: {self.final_val_accuracy:.4f}')

def setup_mlflow_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # parameters for logging
    mlflow.log_param('model', 'VGG16')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('num_classes', NUM_CLASSES)
    mlflow.log_param('input_shape', (224, 224, 3))

    # Final metrics
    mlflow.log_metric('final_val_accuracy', 0)
    mlflow.log_metric('final_val_f1_score', 0)

    # Best metrics
    mlflow.log_metric('best_val_accuracy', 0)
    mlflow.log_metric('best_val_f1_score', 0)
    mlflow.log_metric('best_epoch', 0)
    
    # Epoch metrics
    mlflow.log_metric('train_loss', 0, step=0)
    mlflow.log_metric('train_accuracy', 0, step=0)
    mlflow.log_metric('train_f1_score', 0, step=0)
    mlflow.log_metric('val_accuracy', 0, step=0)
    mlflow.log_metric('val_loss', 0, step=0)
    mlflow.log_metric('val_f1_score', 0, step=0)


################################ production ###########################################
# TODO: Erwin - for production we can use this code to load the best model retreived by get_best_model()
#
# # Get the best model
# model_path, run_id, accuracy = get_best_model() 
# if model_path:
#    # Load the model
#    best_model = mlflow.keras.load_model(model_path)
#    
#    # Copy the model to the production repository or deployment location
#    production_path = "/path/to/production/model"  # path to be defined
#   # Use appropriate methods for model transfer: shutil.copy, git operations, etc.
#    
#    print(f"Best model (accuracy: {accuracy:.4f}) deployed to production")
# else:
#     print("Failed to retrieve the best model")
#########################################################################################

# MAIN FUNCTION FOR TRAINING
def train_model(dataset_path: str = None): 
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)
    
    Arguments:
    - dataset_path: ???

    Returns: None   
    
    '''

    # load mlflow
    setup_mlflow_experiment()
    
    # new insertion
    # TODO: Probably this could be part of the api, the path to the training data?
    train_data, train_records = load_tfrecord_data('data/raw/train_subset6.tfrecord')
    print('Training data loaded ✅')

    val_data, val_records = load_tfrecord_data('data/raw/valid_subset6.tfrecord')
    print('Validation data loaded ✅')

    input_shape = (224, 224, 3)

    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )
    print('Model built ✅')

    # logging in mlflow
    # INFO: Starting MLflow
    mlflow_logger = MLFlowLogger()
    print('MLflow logger started ✅')

    print('Training classification head...', end='\r')
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        callbacks=[mlflow_logger]
    )
    print('Training classification ended ✅')

    # Step 2: Fine-tune the last layers of the base model
    print('Build fine-tuning model...', end='\r')
    tf.keras.backend.clear_session()

    # reinitializing optimizer
    optimizer = optimizers.Adam(learning_rate=1e-4, amsgrad=True)
    
    model, _ = build_vgg16_model(
        input_shape, 
        num_classes, 
        trainable_base=True, 
        fine_tune_layers=4
    )
    print('Fine-tuning model built ✅')

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )

    print('Fine-tuning head...', end='\r')
    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        callbacks=[mlflow_logger]
    )
    print('Fine-tuning ended ✅')

    # saving mlflow loggs
    mlflow.keras.log_model(model, 'model')
    
    # saving locally for comparison
    os.makedirs('temp', exist_ok=True)
    model.save('temp/current_model.keras')
    print('Current model saved under temp/current_model.keras ✅')

    # write current val_accuracy to current_accuracy.txt
    with open('temp/current_accuracy.txt', 'w') as f:
        f.write(str(mlflow_logger.final_val_accuracy))
    print('Saved current final validation accuracy as temp/current_accuracy.txt ✅.')

    mlflow.end_run()
    print('Scores are saved with MLflow ✅.')

    # Combine both training histories
    history = {
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'f1_score': history_1.history['f1_score'] + history_2.history['f1_score'],
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
        'val_f1_score': history_1.history['val_f1_score'] + history_2.history['val_f1_score'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss']
    }

    # Save history as JSON
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
        print(f'History saved in {HISTORY_PATH} ✅.')

    # make git commit for history 
    # Git commit and push
    repo_root = Path.cwd()
    subprocess.run(
        ['git', 'add', str(HISTORY_PATH)], 
        cwd=repo_root, 
        check=True
    )
    commit_msg = 'Updated history logs'
    subprocess.run(
        ['git', 'commit', '-m', commit_msg], 
        cwd=repo_root, 
        check=True
    )
    print('Training completed. ✅')

if __name__ == '__main__':
    train_model()
    result = update_model_if_better()
    print(f'Model management result: {result}')
