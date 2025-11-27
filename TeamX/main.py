"""
Main training script for Fruit Classifier.
Runs the complete pipeline: data loading, training, evaluation, and mislabel audit.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from tensorflow.keras.utils import to_categorical

# Import project modules
from config import DATA_DIR, NUM_CLASSES, SEED, EPOCHS, BATCH_SIZE
from data import load_images_from_flat_directory, preprocess_images
from model import create_cnn_model, create_transfer_learning_model
from train import train_model, save_training_history, create_experiment_directory
from evaluate import generate_evaluation_report
from mislabel_audit import save_mislabel_report
from utils import set_random_seed, log_message


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("FRUIT CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    
    # Set random seed for reproducibility
    log_message("Setting random seed for reproducibility...")
    set_random_seed(SEED)
    
    # Create experiment directory
    log_message("Creating experiment directory...")
    experiment_dir = create_experiment_directory()
    log_message(f"Experiment directory: {experiment_dir}")
    
    # Step 1: Load Training Data
    print("\n" + "=" * 70)
    log_message("STEP 1: Loading training data...")
    print("=" * 70)
    
    train_dir = DATA_DIR / "train"
    X_train, y_train, classes = load_images_from_flat_directory(train_dir)
    log_message(f"Loaded {len(X_train)} training images")
    log_message(f"Classes: {classes}")
    
    # Step 2: Load Test Data
    print("\n" + "=" * 70)
    log_message("STEP 2: Loading test data...")
    print("=" * 70)
    
    test_dir = DATA_DIR / "test"
    X_test, y_test, _ = load_images_from_flat_directory(test_dir)
    log_message(f"Loaded {len(X_test)} test images")
    
    # Step 3: Preprocess Data
    print("\n" + "=" * 70)
    log_message("STEP 3: Preprocessing images...")
    print("=" * 70)
    
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    log_message("Images normalized to [0, 1] range")
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)
    log_message(f"Labels converted to one-hot encoding (shape: {y_train_cat.shape})")
    
    # Step 4: Create Model
    print("\n" + "=" * 70)
    log_message("STEP 4: Creating model...")
    print("=" * 70)
    
    model = create_cnn_model(num_classes=NUM_CLASSES)
    log_message("CNN model created successfully")
    
    print("\nModel Architecture:")
    model.summary()
    
    # Step 5: Train Model
    print("\n" + "=" * 70)
    log_message("STEP 5: Training model...")
    print("=" * 70)
    
    log_message(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    log_message(f"Using test set as validation (not ideal, but works for demo)")
    
    history = train_model(
        model,
        X_train, y_train_cat,
        X_test, y_test_cat,  # Using test as validation
        experiment_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    log_message("Training completed!")
    
    # Step 6: Save Training History
    print("\n" + "=" * 70)
    log_message("STEP 6: Saving training history...")
    print("=" * 70)
    
    save_training_history(history, experiment_dir)
    log_message(f"Training history saved to {experiment_dir / 'history.json'}")
    
    # Step 7: Evaluate Model
    print("\n" + "=" * 70)
    log_message("STEP 7: Evaluating model...")
    print("=" * 70)
    
    log_message("Generating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    log_message("Generating evaluation report...")
    generate_evaluation_report(
        y_test, y_pred, y_pred_proba,
        history,
        experiment_dir
    )
    
    # Step 8: Mislabel Audit
    print("\n" + "=" * 70)
    log_message("STEP 8: Running mislabel audit...")
    print("=" * 70)
    
    suspicious_df, low_conf_df = save_mislabel_report(
        y_test, y_pred, y_pred_proba,
        experiment_dir
    )
    
    # Step 9: Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    
    print(f"\n📊 Results Summary:")
    print(f"   Experiment directory: {experiment_dir}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Number of classes: {NUM_CLASSES}")
    print(f"   Classes: {classes}")
    
    # Calculate final accuracy
    from sklearn.metrics import accuracy_score
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"   ✓ Model: {experiment_dir / 'model_best.h5'}")
    print(f"   ✓ History: {experiment_dir / 'history.json'}")
    print(f"   ✓ Metrics: {experiment_dir / 'metrics.json'}")
    print(f"   ✓ Plots: {experiment_dir / 'plots/'}")
    print(f"   ✓ Mislabel Reports: {experiment_dir / 'mislabels/'}")
    
    print("\n✅ All done! Check the experiment directory for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
