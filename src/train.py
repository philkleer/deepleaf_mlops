import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import json
from data_loader import load_data
from model import build_vgg16_model
from config import MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, PROC_DIR, NUM_CLASSES
import os
from helpers import load_tfrecord_data
import mlflow
import mlflow.keras

# ML Flow setup (still needs to be tested)
class MLFlowLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric("train_loss", logs.get('loss'), step=epoch)
            mlflow.log_metric("train_accuracy", logs.get('accuracy'), step=epoch)
            mlflow.log_metric("val_loss", logs.get('val_loss'), step=epoch)
            mlflow.log_metric("val_accuracy", logs.get('val_accuracy'), step=epoch)

def setup_mlflow_experiment():
    mlflow.set_experiment("Plant_Classification_Experiment")
    mlflow.start_run()
    mlflow.log_param("model", "VGG16")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("input_shape", (224, 224, 3))


# Old function adjusted
def train_model():
    """
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. saves scores with ML Flow (needs to be tested)
    """
    
    # load mlflow
    setup_mlflow_experiment()

    # new insertion
    train_data = load_tfrecord_data(
        os.path.join(
            PROC_DIR,
            "train_data.tfrecord"
        )
    )
    
    val_data = load_tfrecord_data(os.path.join(PROC_DIR, "val_data.tfrecord"))
    
    input_shape = (224, 224, 3)
    # Number of classes from the label shape
    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max'
    )

    # logging in mlflow
    mlflow_logger = MLFlowLogger()

    # manually setting steps per epoch
    steps_per_epoch = sum(1 for _ in train_data)

    print("Training classification head...")
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
    )

    # Step 2: Fine-tune the last layers of the base model
    print("Fine-tuning model...")
    model, _ = build_vgg16_model(
        input_shape, 
        num_classes, 
        trainable_base=True, 
        fine_tune_layers=4
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
    )

    model.save(MODEL_PATH, save_format='keras')
    print(f"Training completed. Model saved at {MODEL_PATH}")

    # saving mlflow logs manually to 
    mlflow.keras.log_model(model, "model")
    mlflow.end_run()
    print("Scores are saved with MLflow.")

    # Combine both training histories
    history = {
        "accuracy": history_1.history["accuracy"] + history_2.history["accuracy"],
        "val_accuracy": history_1.history["val_accuracy"] + history_2.history["val_accuracy"],
        "loss": history_1.history["loss"] + history_2.history["loss"],
        "val_loss": history_1.history["val_loss"] + history_2.history["val_loss"]
    }

    # Save history as JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print(f"Training completed. Model saved at {MODEL_PATH}")
    print(f"Training history saved at {HISTORY_PATH}")


if __name__ == "__main__":
    train_model()

