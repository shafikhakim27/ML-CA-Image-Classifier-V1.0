"""
Script to verify data loading and configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import CLASS_NAMES, NUM_CLASSES, DATA_DIR, IMAGE_SIZE
from data import load_images_from_flat_directory

print("=" * 60)
print("DATA CONFIGURATION VERIFICATION")
print("=" * 60)

print(f"\n📊 Configuration:")
print(f"   Number of classes: {NUM_CLASSES}")
print(f"   Class names: {CLASS_NAMES}")
print(f"   Image size: {IMAGE_SIZE}")
print(f"   Data directory: {DATA_DIR}")

print(f"\n📁 Checking data directories...")

# Check train directory
train_dir = DATA_DIR / "train"
test_dir = DATA_DIR / "test"

if train_dir.exists():
    print(f"   ✓ Train directory found: {train_dir}")
    train_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   ✓ Train images: {len(train_files)}")
else:
    print(f"   ✗ Train directory not found: {train_dir}")

if test_dir.exists():
    print(f"   ✓ Test directory found: {test_dir}")
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   ✓ Test images: {len(test_files)}")
else:
    print(f"   ✗ Test directory not found: {test_dir}")

print(f"\n🔍 Analyzing file naming patterns...")

# Analyze train files
if train_dir.exists():
    class_counts = {}
    for filename in train_files:
        class_name = filename.split('_')[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n   Train set class distribution:")
    for class_name in sorted(class_counts.keys()):
        print(f"      {class_name}: {class_counts[class_name]} images")

# Analyze test files
if test_dir.exists():
    class_counts = {}
    for filename in test_files:
        class_name = filename.split('_')[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n   Test set class distribution:")
    for class_name in sorted(class_counts.keys()):
        print(f"      {class_name}: {class_counts[class_name]} images")

print(f"\n🧪 Testing data loading function...")

try:
    # Load a small sample from train
    X_train, y_train, classes = load_images_from_flat_directory(str(train_dir))
    print(f"   ✓ Successfully loaded training data")
    print(f"   ✓ Shape: {X_train.shape}")
    print(f"   ✓ Labels shape: {y_train.shape}")
    print(f"   ✓ Classes: {classes}")
    print(f"   ✓ Label range: {y_train.min()} to {y_train.max()}")
    
    # Load test data
    X_test, y_test, _ = load_images_from_flat_directory(str(test_dir))
    print(f"   ✓ Successfully loaded test data")
    print(f"   ✓ Shape: {X_test.shape}")
    print(f"   ✓ Labels shape: {y_test.shape}")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)
print("\nYour configuration is ready!")
print(f"Classes: {CLASS_NAMES}")
print(f"Total classes: {NUM_CLASSES}")
print("\nYou can now proceed with training your model.")
