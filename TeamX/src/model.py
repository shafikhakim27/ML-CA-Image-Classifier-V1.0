"""
Model definition and architecture.
Contains the neural network model for fruit classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE, LOSS_FUNCTION, METRICS


def create_cnn_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Create a simple CNN model for image classification.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    compile_model(model)
    return model


def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, trainable=False):
    """
    Create a transfer learning model using MobileNetV2.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        trainable: Whether to train the base model
        
    Returns:
        Compiled Keras model
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    compile_model(model)
    return model


def compile_model(model, optimizer=None, loss=LOSS_FUNCTION, metrics=METRICS):
    """
    Compile the model with specified optimizer and loss.
    
    Args:
        model: Keras model to compile
        optimizer: Optimizer to use (default: Adam)
        loss: Loss function
        metrics: Metrics to track
    """
    if optimizer is None:
        optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def print_model_summary(model):
    """Print model architecture summary."""
    model.summary()
