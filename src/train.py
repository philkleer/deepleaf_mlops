import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import json
from data_loader import load_data
from model import build_vgg16_model
from config import MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, PROC_DIR, NUM_CLASSES
import os
from helpers import load_tfrecord_data

# Old function adjusted
def train_model():
    """
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    """
    # train_data, val_data = load_data()
    
    # new insertion
    train_data = load_tfrecord_data(
        os.path.join(
            PROC_DIR,
            "train_data.tfrecord"
        )
    )
    
    val_data = load_tfrecord_data(os.path.join(PROC_DIR, "val_data.tfrecord"))
    
    input_shape = (224, 224, 3)
    # num_classes = train_data.num_classes
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

    # manually setting steps per epoch
    steps_per_epoch = sum(1 for _ in train_data)

    print("Training classification head...")
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint]
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
        callbacks=[checkpoint]
    )

    model.save(MODEL_PATH, save_format='keras')
    print(f"Training completed. Model saved at {MODEL_PATH}")

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

