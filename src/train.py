import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras import optimizers, callbacks
import json
from model import build_vgg16_model
from config import MODEL_DIR, MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES
import os
from helpers import load_tfrecord_data
import mlflow
import mlflow.keras
from dotenv import load_dotenv
from datetime import datetime

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

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_KEY')

# Save better model
def save_model_if_better(current_model, current_val_accuracy, model_path):
    """
    Compare the current model's validation accuracy with the best from all MLflow runs
    and save only if it's better.
    """
    best_val_accuracy = get_best_val_accuracy()  # Fetch previous best accuracy

    if current_val_accuracy > best_val_accuracy:
        current_model.save(model_path, save_format='keras')
        print(f"🔥 New best model saved! Validation accuracy improved from {best_val_accuracy} → {current_val_accuracy}")
    else:
        print(f"⚠️ No improvement. Best accuracy remains {best_val_accuracy}. Model not saved.")


# get best val accuracy
def get_best_val_accuracy():
    """Fetch the best validation accuracy from all previous MLflow runs."""
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("Plant_Classification_Experiment")
    
    if experiment is None:
        return 0   
    
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id, order_by=["metrics.best_val_accuracy DESC"], max_results=1)
    
    if runs:
        return runs[0].data.metrics.get("best_val_accuracy", 0)  # Get highest accuracy run
    return 0  

# ML Flow setup
class MLFlowLogger(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.final_val_accuracy = 0
        self.final_val_f1_score = 0
        self.final_run_id = None
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # training scores
            mlflow.log_metric('train_loss', logs.get('loss'), step=epoch)
            mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=epoch)
            mlflow.log_metric('train_f1_score', logs.get('f1_score'), step=epoch)
            # validation scores
            mlflow.log_metric('val_loss', logs.get('val_loss'), step=epoch)
            mlflow.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch)
            mlflow.log_metric('val_f1_score', logs.get('val_f1_score'), step=epoch)

        # here we check if the recent val_accuracy is the best 
        val_accuracy = logs.get('val_accuracy')
        
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            mlflow.log_metric('best_val_accuracy', self.best_val_accuracy)
            mlflow.log_metric('best_val_f1_score', self.best_val_f1_score)
            print(f'Updated best validation accuracy: {round(val_accuracy, 4)} ✅')

    def on_train_end(self, logs=None):        
        print(f'Final validation accuracy: {round(logs.get("val_accuracy", 0), 4)}')
        # Log the final results, this is what you'll compare across models
        mlflow.log_metric('final_val_accuracy', logs.get('val_accuracy', 0))
        mlflow.log_metric('final_val_f1_score', logs.get('val_f1_score', 0))

def setup_mlflow_experiment():
    mlflow.set_tracking_uri('https://dagshub.com/philkleer/deepleaf_mlops.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')

    # parameters for logging
    mlflow.log_param('model', 'VGG16')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('num_classes', NUM_CLASSES)
    mlflow.log_param('input_shape', (224, 224, 3))
    
    # Final metrics
    final_val_accuracy = float(mlflow.active_run().data.metrics.get('final_val_accuracy', 0))
    final_val_f1_score = float(mlflow.active_run().data.metrics.get('final_val_f1_score', 0))

    mlflow.log_metric('final_val_accuracy', final_val_accuracy)
    mlflow.log_metric('final_val_f1_score', final_val_f1_score)

    # Best metrics
    best_val_accuracy = float(mlflow.active_run().data.metrics.get('best_val_accuracy', 0))
    best_val_f1_score = float(mlflow.active_run().data.metrics.get('best_val_f1_score', 0))

    mlflow.log_metric('best_val_accuracy', best_val_accuracy)
    mlflow.log_metric('best_val_f1_score', best_val_f1_score)

    # epoch metrics
    mlflow.log_metric('train_accuracy', 0, step=0)
    mlflow.log_metric('train_loss', 0, step=0)
    mlflow.log_metric('train_f1_score', 0, step=0)
    mlflow.log_metric('val_accuracy', 0, step=0)
    mlflow.log_metric('val_loss', 0, step=0)
    mlflow.log_metric('val_f1_score', 0, step=0)

def train_model():
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores 
    '''
        
    # load mlflow
    setup_mlflow_experiment()
    
    # new insertion
    # TODO: Probably this could be part of the api, the path to the training data?
    train_data, train_records = load_tfrecord_data('data/raw/train_subset1.tfrecord')
    print('Training data loaded ✅')

    val_data, val_records = load_tfrecord_data('data/raw/valid_subset1.tfrecord')
    print('Validation data loaded ✅')

    input_shape = (224, 224, 3)

    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(
        input_shape,
        num_classes, 
        trainable_base=False
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=0.0001,
            # clipvalue=1
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )
    print('Training model built ✅')

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
    print('Fine-tuning model...', end='\r')
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
        metrics=['accuracy', F1Score()]
    )

    print('Fine-Tuning classification head...', end='\r')
    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        callbacks=[mlflow_logger]
    )
    print('Fine-Tuning classification ended ✅')

    # saving mlflow loggs
    mlflow.keras.log_model(model, 'model')
    mlflow.end_run()
    print('Scores are saved with MLflow ✅.')

    # Combine both training histories
    history = {
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss'],
        'f1_score': history_1.history['f1_score'] + history_2.history['f1_score'],
        'val_f1_score': history_1.history['val_f1_score'] + history_2.history['val_f1_score']
    }

    # Save history as JSON
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
        print(f'History saved in {HISTORY_PATH} ✅.')

    # saving model
    final_val_accuracy = history_2.history['val_accuracy'][-1] 

    save_model_if_better(model, final_val_accuracy, MODEL_PATH)

    # saving model per se
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR_TIMESTAMP = f"{MODEL_DIR}/model_{timestamp}.keras"

    model.save(MODEL_DIR_TIMESTAMP, save_format='keras')
    print(f'Model saved under {MODEL_DIR_TIMESTAMP} ✅')
    print('Training completed. 🏁')

if __name__ == '__main__':
    train_model()

