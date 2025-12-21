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


def create_data_augmentation_generators(target_size=(150, 150), augmentation_config=None):
    """
    Create data augmentation and test generators.
    OPTIMIZED for 92%+ accuracy with enhanced augmentation.
    
    Args:
        target_size: Target image size (height, width). Default 150x150 (optimized from 224)
        augmentation_config: Dictionary with augmentation parameters (from config.AUGMENTATION_CONFIG)
        
    Returns:
        train_datagen: Generator for training data with augmentation
        test_datagen: Generator for test data (no augmentation)
    """
    from config import AUGMENTATION_CONFIG as default_aug
    if augmentation_config is None:
        augmentation_config = default_aug
    
    # --- 1. DATA AUGMENTATION (OPTIMIZED) ---
    # Enhanced augmentation for 92%+ accuracy
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=augmentation_config.get('rotation_range', 45),
        width_shift_range=augmentation_config.get('width_shift_range', 0.25),
        height_shift_range=augmentation_config.get('height_shift_range', 0.25),
        shear_range=augmentation_config.get('shear_range', 0.25),
        zoom_range=augmentation_config.get('zoom_range', 0.3),
        horizontal_flip=augmentation_config.get('horizontal_flip', True),
        vertical_flip=augmentation_config.get('vertical_flip', True),  # NEW
        brightness_range=augmentation_config.get('brightness_range', [0.7, 1.3]),
        channel_shift_range=augmentation_config.get('channel_shift_range', 30),
        fill_mode=augmentation_config.get('fill_mode', 'reflect'),  # OPTIMIZED: reflect
        validation_split=0.2
    )
    
    # Test data should NOT be augmented, only scaled.
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen


def load_data_generators(train_dir='data/train', test_dir='data/test', 
                        target_size=(224, 224), batch_size=16):
    """
    Load data using flow_from_directory with augmentation.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        target_size: Target image size (default 224x224 for MobileNetV2)
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


def create_transfer_learning_model(input_shape=(150, 150, 3), num_classes=NUM_CLASSES, fine_tune=True):
    """
    Create OPTIMIZED transfer learning model using MobileNetV2 for 92%+ accuracy.
    
    MobileNetV2 is ideal for small datasets because:
    - Pre-trained on 1.3M ImageNet images
    - Fine-tuning last 15 layers for fruit classification
    - Enhanced regularization: L2 + Increased Dropout
    - Expected improvement: 91.67% → 92%+
    
    Args:
        input_shape: Shape of input images (default 150x150x3 - OPTIMIZED)
        num_classes: Number of output classes
        fine_tune: Whether to fine-tune base layers (default True - OPTIMIZED)
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from config import REGULARIZATION_CONFIG
    
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # OPTIMIZED: Fine-tune last 15 layers
    if fine_tune:
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        for layer in base_model.layers[-15:]:
            layer.trainable = True
    else:
        base_model.trainable = False
    
    # Get regularization config
    reg_cfg = REGULARIZATION_CONFIG
    l2_reg = l2(reg_cfg.get('l2_value', 0.0001)) if reg_cfg.get('use_l2') else None
    
    # OPTIMIZED: Enhanced architecture with L2 + improved dropout
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(reg_cfg.get('dropout_1', 0.6)))
    model.add(Dense(reg_cfg['dense_units'][0], activation='relu', kernel_regularizer=l2_reg))
    if reg_cfg.get('use_batch_norm'):
        model.add(BatchNormalization())
    model.add(Dropout(reg_cfg.get('dropout_2', 0.5)))
    model.add(Dense(reg_cfg['dense_units'][1], activation='relu', kernel_regularizer=l2_reg))
    if reg_cfg.get('use_batch_norm'):
        model.add(BatchNormalization())
    model.add(Dropout(reg_cfg.get('dropout_3', 0.4)))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile with optimized learning rate
    compile_model(model, optimizer=Adam(learning_rate=LEARNING_RATE))
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
