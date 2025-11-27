"""
Model definition and architecture.
Contains the neural network model for fruit classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, GaussianNoise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE, LOSS_FUNCTION, METRICS


def create_data_augmentation_generators(target_size=(150, 150)):
    """
    Create data augmentation and test generators.
    
    Args:
        target_size: Target image size (height, width)
        
    Returns:
        train_datagen: Generator for training data with augmentation
        test_datagen: Generator for test data (no augmentation)
    """
    # --- 1. DATA AUGMENTATION (The "Confusion" Generator) ---
    # This creates new variations of your photos on the fly.
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values
        rotation_range=40,                 # Tilt photo up to 40 degrees
        width_shift_range=0.2,             # Shift left/right
        height_shift_range=0.2,            # Shift up/down
        shear_range=0.2,                   # Distort shape (shear)
        zoom_range=0.2,                    # Zoom in/out
        horizontal_flip=True,              # Mirror image
        brightness_range=[0.8, 1.2],       # Simulate different lighting
        channel_shift_range=20.0,          # Slight color changes
        fill_mode='nearest'
    )
    
    # Test data should NOT be augmented, only scaled.
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen


def load_data_generators(train_dir='data/train', test_dir='data/test', 
                        target_size=(150, 150), batch_size=16):
    """
    Load data using flow_from_directory with augmentation.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        target_size: Target image size
        batch_size: Batch size for generators
        
    Returns:
        train_generator: Training data generator
        test_generator: Test data generator
    """
    train_datagen, test_datagen = create_data_augmentation_generators(target_size)
    
    # Load Data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, test_generator


def compute_class_weights(train_generator):
    """
    Compute class weights to handle imbalanced data.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        class_weight_dict: Dictionary of class weights
    """
    # --- 2. BALANCING (Handling unequal amounts of data) ---
    # If one fruit has fewer samples, this penalizes mistakes on that class more.
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    # Convert to dictionary format required by Keras
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict


def create_cnn_model(input_shape=(150, 150, 3), num_classes=NUM_CLASSES):
    """
    Create a simplified CNN model with max 2 convolutional layers.
    Includes data augmentation noise and dropout for regularization.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # --- 3. ARCHITECTURE (Add Noise, Remove Complexity) ---
    model = Sequential()
    
    # Input Layer + Gaussian Noise (Artificial Static)
    model.add(Input(shape=input_shape))
    # This adds random noise to training data to prevent memorization
    model.add(GaussianNoise(0.1))
    
    # Convolution Block 1
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Convolution Block 2 (Max 2 convolutional layers)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    
    # Dropout: Randomly sets 50% of inputs to 0.
    # This forces the model to not rely on specific paths.
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
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
