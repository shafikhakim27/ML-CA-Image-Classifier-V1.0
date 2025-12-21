"""
Training module for the fruit classifier model.
Handles model training, validation, and checkpointing.
"""

import json
from pathlib import Path
from datetime import datetime
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from config import (
    EPOCHS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    LEARNING_RATE, BATCH_SIZE, AUGMENTATION_CONFIG,
    EXPERIMENTS_DIR, SEED
)
import numpy as np


def setup_callbacks(experiment_dir, patience=EARLY_STOPPING_PATIENCE):
    """
    Setup training callbacks - ENHANCED for 92%+ accuracy.
    
    ENHANCEMENT CHANGES:
    - Early stopping patience: 10 → 15 epochs (longer training window)
    - LR reduction factor: 0.3 → 0.2 (more aggressive LR decay)
    - LR minimum: 1e-8 → 1e-9 (even lower floor for fine-tuning)
    - Early stopping min_delta: 0.002 → 0.001 (stricter improvement threshold)
    - LR reduction patience: 3 → 4 (wait longer before reducing)
    
    Args:
        experiment_dir: Directory to save checkpoints
        patience: Patience for early stopping (default 15 from config)
        
    Returns:
        List of callbacks
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,  # ENHANCED: 15 epochs (was 10)
            min_delta=EARLY_STOPPING_MIN_DELTA,  # ENHANCED: 0.001 (was 0.002)
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,  # ENHANCED: 0.2 (was 0.3 - more aggressive)
            patience=REDUCE_LR_PATIENCE,  # ENHANCED: 4 (was 3)
            min_lr=REDUCE_LR_MIN,  # ENHANCED: 1e-9 (was 1e-8)
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(experiment_dir / 'model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val,
                experiment_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight_dict=None):
    """
    Train the model - ENHANCED for 92%+ accuracy.
    
    ENHANCEMENT CHANGES:
    - Epochs: 100 → 150 (extended training for better convergence)
    - Batch size: 32 → 16 (more gradient updates per epoch)
    - Learning rate: 0.001 → 0.00005 (even more conservative for fine-tuning)
    
    Args:
        model: Keras model to train
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        experiment_dir: Directory to save outputs
        epochs: Number of epochs (default 150 from config)
        batch_size: Batch size (default 16 from config)
        class_weight_dict: Class weights for imbalanced data
        
    Returns:
        Training history
    """
    callbacks = setup_callbacks(experiment_dir)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return history


def train_model_with_generators(model, train_generator, test_generator,
                               experiment_dir, epochs=EPOCHS, class_weight_dict=None):
    """
    Train the model using data generators (with augmentation).
    
    Args:
        model: Keras model to train
        train_generator: Training data generator
        test_generator: Test/validation data generator
        experiment_dir: Directory to save outputs
        epochs: Number of epochs
        class_weight_dict: Class weights for handling imbalanced data
        
    Returns:
        Training history
    """
    callbacks = setup_callbacks(experiment_dir)
    
    print(f"Starting training with Class Weights: {class_weight_dict}")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return history


def save_training_history(history, experiment_dir):
    """
    Save training history to JSON file.
    
    Args:
        history: Training history object
        experiment_dir: Directory to save to
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    history_dict = {k: [float(v) if isinstance(v, np.floating) else v
                        for v in history.history[k]]
                    for k in history.history.keys()}
    
    with open(experiment_dir / 'history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)


def create_experiment_directory(base_exp_num=1):
    """
    Create a new experiment directory with incremented number.
    
    Args:
        base_exp_num: Starting experiment number
        
    Returns:
        Path to new experiment directory
    """
    exp_counter = base_exp_num
    while True:
        exp_dir = EXPERIMENTS_DIR / f"exp_{exp_counter:03d}_baseline"
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir
        exp_counter += 1
