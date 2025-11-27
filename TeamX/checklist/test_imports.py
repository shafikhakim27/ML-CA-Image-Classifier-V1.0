"""
Test script to verify all package imports work correctly.
"""

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✓ seaborn imported successfully")
except ImportError as e:
    print(f"✗ seaborn import failed: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")

try:
    import tensorflow as tf
    print(f"✓ tensorflow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"✗ tensorflow import failed: {e}")

try:
    import keras
    print(f"✓ keras {keras.__version__} imported successfully")
except ImportError as e:
    print(f"✗ keras import failed: {e}")

try:
    import cv2
    print("✓ opencv-python imported successfully")
except ImportError as e:
    print(f"✗ opencv-python import failed: {e}")

try:
    from PIL import Image
    print("✓ Pillow imported successfully")
except ImportError as e:
    print(f"✗ Pillow import failed: {e}")

try:
    from tqdm import tqdm
    print("✓ tqdm imported successfully")
except ImportError as e:
    print(f"✗ tqdm import failed: {e}")

print("\nTesting project module imports...")

try:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    from config import IMAGE_SIZE, BATCH_SIZE
    print(f"✓ config imported successfully (IMAGE_SIZE={IMAGE_SIZE})")
    
    from model import create_cnn_model
    print("✓ model imported successfully")
    
    from data import load_images_from_directory
    print("✓ data imported successfully")
    
    from train import setup_callbacks
    print("✓ train imported successfully")
    
    from evaluate import calculate_metrics
    print("✓ evaluate imported successfully")
    
    from mislabel_audit import identify_potential_mislabels
    print("✓ mislabel_audit imported successfully")
    
    from utils import create_directories
    print("✓ utils imported successfully")
    
except ImportError as e:
    print(f"✗ Project module import failed: {e}")

print("\n✅ All import tests completed!")
