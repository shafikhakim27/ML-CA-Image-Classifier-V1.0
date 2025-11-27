"""
Utility functions for the fruit classifier project.
Contains helper functions for logging, file operations, and common tasks.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np


def create_directories(directories):
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_json(file_path):
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def log_message(message, log_file=None):
    """
    Log message with timestamp.
    
    Args:
        message: Message to log
        log_file: Optional file to log to
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')


def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_file_size(file_path):
    """
    Get file size in MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return os.path.getsize(file_path) / (1024 * 1024)


def list_files_in_directory(directory, extension=None):
    """
    List files in a directory.
    
    Args:
        directory: Path to directory
        extension: Optional file extension filter
        
    Returns:
        List of file paths
    """
    files = []
    for item in Path(directory).iterdir():
        if item.is_file():
            if extension is None or item.suffix == extension:
                files.append(item)
    return sorted(files)
