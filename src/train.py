import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import json
from model import build_vgg16_model
from config import MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES
import os
from helpers import load_tfrecord_data
import mlflow
import mlflow.keras

# ML Flow setup (still needs to be tested)
class MLFlowLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric('train_loss', logs.get('loss'), step=epoch)
            mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=epoch)
            mlflow.log_metric('val_loss', logs.get('val_loss'), step=epoch)
            mlflow.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch)

            # checking if it is the best epoch based on validation
            if logs.get('val_accuracy') > mlflow.active_run().data.params.get('best_val_accuracy', 0):
                mlflow.log_param('best epoch', epoch)
                mlflow.log_param('best_val_accuracy', logs.get('val_accuracy'))

                # saving this model (in the first run, this happens after each epoch probably, in the following runs it shouldn't)
                # DEBUG: Erwin, I did this saving here, where the values are also saved 
                self.model.save(MODEL_PATH, save_format='keras')

def setup_mlflow_experiment():
    # TODO: set up later after Yannick created dagshub
    # DEBUG: Yannik, here you have to add the repository name
    mlflow.set_tracking_uri('https://dagshub.com/<username>/<repo_name>.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')
    mlflow.start_run()

    # parameters for logging
    mlflow.log_param('model', 'VGG16')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('num_classes', NUM_CLASSES)
    mlflow.log_param('input_shape', (224, 224, 3))

    # Getting the so far best score, if there are runs
    # INFO: I just implemented this without further checking
    # DEBUG: Erwin, this would be where could implement your approach to get the best value so far for the comparison later on  
    best_run = mlflow.search_runs(order_by=["val_accuracy desc"]).head(1)
    if len(best_run) > 0:
        best_epoch = best_run.iloc[0]['best_epoch']
        best_val_accuracy = best_run.iloc[0]['best_val_accuracy']
        mlflow.log_param('best_val_accuracy', best_val_accuracy)

    else:
        mlflow.log_param('best_val_accuracy', 0)

# TODO: Creatin function to get best epoch (accuracy):
# DEBUG: Erwin, here should be the function to make the comparison
def get_best_epoch_and_accuracy():
    # Fetching the best run from MLflow
    best_run = mlflow.search_runs(order_by=["val_accuracy desc"]).head(1)
    if len(best_run) > 0:
        best_epoch = best_run.iloc[0]['best_epoch']
        best_val_accuracy = best_run.iloc[0]['best_val_accuracy']
        return best_epoch, best_val_accuracy
    else:
        return None, None

# Old function adjusted
def train_model():
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)
    '''
    # INFO: data is not loaded again, since it will now load the `.tfrecord` files
    # train_data, val_data = load_data()
    
    # load mlflow
    setup_mlflow_experiment()
    
    # new insertion
    # TODO: Probably this could be part of the api, the path to the training data?
    train_data = load_tfrecord_data('data/raw/train_subset1.tfrecord')
    print('Training data loaded ✅')

    val_data = load_tfrecord_data('data/raw/valid_subset1.tfrecord')
    print('Validation data loaded ✅')

    input_shape = (224, 224, 3)

    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print('Model built ✅')

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max'
    )

    # logging in mlflow
    # INFO: Starting MLflow
    mlflow_logger = MLFlowLogger()
    print('MLflow logger started ✅')

    # manually setting steps per epoch
    steps_per_epoch = sum(1 for _ in train_data)

    print('Training classification head...', end='\r')
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
    )
    print('Training classification ended ✅')


    # Step 2: Fine-tune the last layers of the base model
    print('Fine-tuning model...', end='\r')
    model, _ = build_vgg16_model(
        input_shape, 
        num_classes, 
        trainable_base=True, 
        fine_tune_layers=4
    )
    print('Fine-tuning model ended ✅')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('Training classification head...', end='\r')
    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
    )
    print('Training classification ended ✅')

    # saving mlflow loggs
    mlflow.keras.log_model(model, 'model')
    mlflow.end_run()
    print('Scores are saved with MLflow ✅.')

    # Combine both training histories
    history = {
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss']
    }

    # Save history as JSON
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
        print(f'History saved in {HISTORY_PATH} ✅.')

    print(f'Training completed. Model saved at {MODEL_PATH} ✅')

if __name__ == '__main__':
    train_model()

