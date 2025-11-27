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

# Model configuration
MODEL_NAME = "fruit_classifier"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data configuration
SEED = 42
NUM_CLASSES = 4  # apple, banana, orange, mixed
CLASS_NAMES = ["apple", "banana", "mixed", "orange"]  # Alphabetically sorted class names

# Training configuration
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = ["accuracy", "precision", "recall"]

# Evaluation configuration
CONFIDENCE_THRESHOLD = 0.7
MISLABEL_THRESHOLD = 0.5  # Confidence threshold for flagging potential mislabels

# Output configuration
SAVE_PLOTS = True
SAVE_METRICS = True
VERBOSE = 1
