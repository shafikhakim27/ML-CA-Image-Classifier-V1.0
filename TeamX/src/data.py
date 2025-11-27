"""
Data loading and preprocessing module.
Handles loading, splitting, and augmenting image data.
Uses relative paths via config.py for cross-platform compatibility.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    DATA_DIR, SEED, BATCH_SIZE, IMAGE_SIZE,
    VALIDATION_SPLIT, TEST_SPLIT, CLASS_NAMES
)


def load_images_from_flat_directory(data_dir, image_size=IMAGE_SIZE, class_names=CLASS_NAMES):
    """
    Load images from flat directory where filenames indicate class.
    Expected structure: data/classname_number.jpg (e.g., apple_1.jpg, banana_2.jpg)
    
    Args:
        data_dir: Path to data directory
        image_size: Target image size
        class_names: List of class names to use for labeling
        
    Returns:
        X: Array of images
        y: Array of labels
        classes: List of class names
    """
    from tensorflow.keras.preprocessing import image
    import os
    
    X, y = [], []
    
    # Create a mapping from class name to index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Load all image files
    for img_file in sorted(os.listdir(data_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        try:
            # Extract class name from filename (e.g., "apple_1.jpg" -> "apple")
            class_name = img_file.split('_')[0]
            
            if class_name not in class_to_idx:
                print(f"Warning: Unknown class '{class_name}' in file {img_file}, skipping")
                continue
            
            # Load and process image
            img_path = os.path.join(data_dir, img_file)
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            
            X.append(img_array)
            y.append(class_to_idx[class_name])
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    print(f"Loaded {len(X)} images from {data_dir}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return np.array(X), np.array(y), class_names


def preprocess_images(X, normalize=True):
    """
    Preprocess images for model input.
    
    Args:
        X: Array of images
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image array
    """
    if normalize:
        X = X / 255.0
    return X


def split_data(X, y, test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=SEED):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array
        y: Label array
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Create data generators for training with augmentation.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size
        
    Returns:
        train_generator, val_generator
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False
    )
    
    return train_generator, val_generator
