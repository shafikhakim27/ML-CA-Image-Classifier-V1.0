"""
Configuration module for Fruit Classifier project.
Contains all hyperparameters and settings used across the project.
Works with both relative imports and Jupyter notebooks.
"""

import os
from pathlib import Path

# Project paths - Works whether run from src/ or from root
# When imported as module: uses parent of src/ directory
# When run in Jupyter from src/: looks for data/ in current directory
try:
    PROJECT_ROOT = Path(__file__).parent.parent  # For module imports
except (NameError, AttributeError):
    PROJECT_ROOT = Path.cwd()  # For Jupyter notebooks

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORT_DIR = PROJECT_ROOT / "report"

# Model configuration (ENHANCED for 92%+ accuracy)
MODEL_NAME = "fruit_classifier_optimized_92plus"
IMAGE_SIZE = (150, 150)  # OPTIMIZED: 150x150 (was 224x224) - matches MobileNetV2 training
BATCH_SIZE = 16  # OPTIMIZED: 16 (was 32) - more gradient updates per epoch
EPOCHS = 150  # ENHANCED: 150 (was 100) - extended training for convergence
LEARNING_RATE = 0.00005  # ENHANCED: 0.00005 (was 0.0001) - even more conservative for fine-tuning
FINE_TUNE_LAYERS = 20  # ENHANCED: 20 (was 15) - more layers trainable for better adaptation
DENSE_LAYER_SIZES = [768, 384, 128]  # ENHANCED: Larger intermediate layers for feature extraction
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
USE_FINE_TUNING = True  # Enable fine-tuning of base model layers

# Data configuration
SEED = 42
NUM_CLASSES = 4  # apple, banana, orange, mixed
CLASS_NAMES = ["apple", "banana", "mixed", "orange"]  # Alphabetically sorted class names

# Training configuration (ENHANCED for 92%+ accuracy)
EARLY_STOPPING_PATIENCE = 15  # ENHANCED: 15 (was 10) - allow even more epochs for convergence
EARLY_STOPPING_MIN_DELTA = 0.001  # ENHANCED: 0.001 (was 0.002) - stricter improvement threshold
REDUCE_LR_PATIENCE = 4  # ENHANCED: 4 (was 3) - wait longer before reducing LR
REDUCE_LR_FACTOR = 0.2  # ENHANCED: 0.2 (was 0.3) - more aggressive LR reduction
REDUCE_LR_MIN = 1e-9  # ENHANCED: 1e-9 (was 1e-8) - can go even lower for fine-tuning
OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy", "precision", "recall"]
USE_CLASS_WEIGHTS = True  # Enable class balancing

# Data augmentation configuration (ENHANCED for 92%+ accuracy)
AUGMENTATION_CONFIG = {
    'rotation_range': 50,          # ENHANCED: 50° (was 45°)
    'width_shift_range': 0.3,      # ENHANCED: 30% (was 25%)
    'height_shift_range': 0.3,     # ENHANCED: 30% (was 25%)
    'shear_range': 0.3,            # ENHANCED: 30° (was 25%)
    'zoom_range': 0.4,             # ENHANCED: 40° (was 30%)
    'horizontal_flip': True,
    'vertical_flip': True,         # Enable vertical flips for variety
    'brightness_range': [0.6, 1.4],  # ENHANCED: 0.6-1.4x (was 0.7-1.3x) - wider range
    'channel_shift_range': 40.0,     # ENHANCED: 40 (was 30)
    'fill_mode': 'reflect'           # Better edge handling vs 'nearest'
}

# Evaluation configuration
CONFIDENCE_THRESHOLD = 0.7
MISLABEL_THRESHOLD = 0.5  # Confidence threshold for flagging potential mislabels

# Model regularization configuration (ENHANCED for 92%+ accuracy)
REGULARIZATION_CONFIG = {
    'dropout_1': 0.7,      # ENHANCED: 0.7 (was 0.6) - more aggressive
    'dropout_2': 0.6,      # ENHANCED: 0.6 (was 0.5) - more aggressive
    'dropout_3': 0.5,      # ENHANCED: 0.5 (was 0.4) - more aggressive
    'dropout_4': 0.3,      # NEW: Additional dropout layer after Dense(128)
    'use_l2': True,        # L2 regularization on all dense layers
    'l2_value': 0.0001,    # L2 regularization strength
    'use_batch_norm': True,  # Batch normalization after each dense layer
    'dense_units': [768, 384, 128]  # ENHANCED: Larger intermediate layers (was 512, 256)
}

# Output configuration
SAVE_PLOTS = True
SAVE_METRICS = True
VERBOSE = 1
