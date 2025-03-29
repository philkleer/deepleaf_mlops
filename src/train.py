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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # Sends logs to stdout
)
logger = logging.getLogger(__name__)

# imports from other scripts
from src.config import HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES, MLFLOW_TRACKING_URL
from src.model import build_vgg16_model
from src.helpers import load_tfrecord_data
from src.prod_model_select import update_model_if_better
from src.data_loader import create_data

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

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

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
        logger.info(f'Best Validation Accuracy: {self.best_val_accuracy:.4f}')
        logger.info(f'Final validation accuracy: {self.final_val_accuracy:.4f}')

def setup_mlflow_experiment():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(
        os.getenv('MLFLOW_EXPERIMENT_NAME', 'Plant_Classification_Experiment')
    )

    logger.info(f'MLflow tracking URI set to: {os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5001")}')
    logger.info(f'MLflow experiment set to: {os.getenv("MLFLOW_EXPERIMENT_NAME", "Plant_Classification_Experiment")}')


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

# MAIN FUNCTION FOR TRAINING
def train_model(): 
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)

    Returns: None   
    
    '''

    # load mlflow
    setup_mlflow_experiment()
    
    # new insertion
    train_data, train_records = load_tfrecord_data(
        'data/training/train.tfrecord'
    )
    logger.info('Training data loaded ✅')

    val_data, val_records = load_tfrecord_data(
        'data/training/valid.tfrecord'
    )
    logger.info('Validation data loaded ✅')

    input_shape = (224, 224, 3)

    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )
    logger.info('Model built ✅')

    # logging in mlflow
    # Starting MLflow
    mlflow_logger = MLFlowLogger()
    logger.info('MLflow logger started ✅')

    logger.info('Training classification head...')
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        callbacks=[mlflow_logger]
    )
    logger.info('Training classification ended ✅')

    # Step 2: Fine-tune the last layers of the base model
    logger.info('Build fine-tuning model...')
    tf.keras.backend.clear_session()

    # reinitializing optimizer
    optimizer = optimizers.Adam(learning_rate=1e-4, amsgrad=True)
    
    model, _ = build_vgg16_model(
        input_shape, 
        num_classes, 
        trainable_base=True, 
        fine_tune_layers=4
    )
    logger.info('Fine-tuning model built ✅')

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )

    logger.info('Fine-tuning head...')
    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        callbacks=[mlflow_logger]
    )
    logger.info('Fine-tuning ended ✅')

    # saving mlflow loggs
    mlflow.keras.log_model(model, 'model')
    
    # saving locally for comparison
    os.makedirs('temp', exist_ok=True)
    model.save('temp/current_model.keras')
    logger.info('Current model saved under temp/current_model.keras ✅')

    # write current val_accuracy to current_accuracy.txt
    with open('temp/current_accuracy.txt', 'w') as f:
        f.write(str(mlflow_logger.final_val_accuracy))
    logger.info('Saved current final validation accuracy as temp/current_accuracy.txt ✅.')

    mlflow.end_run()
    logger.info('Scores are saved with MLflow ✅.')

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

    logger.info('Training completed. ✅')

if __name__ == '__main__':
    create_data()
    train_model()
    result = update_model_if_better()
    print(f'Model management result: {result}')
