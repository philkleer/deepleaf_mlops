import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_model(input_shape, num_classes, trainable_base=False, fine_tune_layers=0):
    """
    Builds a transfer learning model using VGG16 as the base model with a custom classification head.

    Args:
    - input_shape: Tuple, shape of input images (we use (224, 224, 3) as this is the shape of the ImageNet data).
    - num_classes: Integer, number of output classes.
    - trainable_base: Boolean, whether to unfreeze the base model.
    - fine_tune_layers: Integer, number of last layers to fine-tune.

    Returns:
    - model: A compiled Keras model.
    - base_model: The VGG16 base model.
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False, 
        input_shape=input_shape
    )
    
    base_model.trainable = trainable_base  # Freeze or unfreeze base layers

    if trainable_base and fine_tune_layers > 0:
        for layer in base_model.layers[:-fine_tune_layers]:  # Unfreeze last few layers
            layer.trainable = False

    # Our custom classification part on top of the base model
    regularizer = tf.keras.regularizers.l2(1e-4)

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

