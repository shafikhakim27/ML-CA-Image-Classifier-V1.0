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
    EPOCHS, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    EXPERIMENTS_DIR, SEED
)
import numpy as np


def setup_callbacks(experiment_dir, patience=EARLY_STOPPING_PATIENCE):
    """
    Setup training callbacks.
    
    Args:
        experiment_dir: Directory to save checkpoints
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
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
                experiment_dir, epochs=EPOCHS, batch_size=32):
    """
    Train the model.
    
    Args:
        model: Keras model to train
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        experiment_dir: Directory to save outputs
        epochs: Number of epochs
        batch_size: Batch size
        
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
