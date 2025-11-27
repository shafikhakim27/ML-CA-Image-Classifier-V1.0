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
from model import (
    create_cnn_model, create_transfer_learning_model,
    load_data_generators, compute_class_weights
)
from train import train_model, train_model_with_generators, save_training_history, create_experiment_directory
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
    
    # Step 1: Load Data Generators (with Augmentation)
    print("\n" + "=" * 70)
    log_message("STEP 1: Loading data with augmentation generators...")
    print("=" * 70)
    
    train_generator, test_generator = load_data_generators(
        train_dir='../data/train',
        test_dir='../data/test',
        target_size=(150, 150),
        batch_size=BATCH_SIZE
    )
    log_message(f"Training samples per batch: {train_generator.batch_size}")
    log_message(f"Total training batches: {len(train_generator)}")
    log_message(f"Total test batches: {len(test_generator)}")
    
    # Step 2: Compute Class Weights (for imbalanced data)
    print("\n" + "=" * 70)
    log_message("STEP 2: Computing class weights for imbalanced data...")
    print("=" * 70)
    
    class_weight_dict = compute_class_weights(train_generator)
    log_message(f"Class weights: {class_weight_dict}")
    
    # Step 3: Create Model
    print("\n" + "=" * 70)
    log_message("STEP 3: Creating model (max 2 convolutional layers)...")
    print("=" * 70)
    
    model = create_cnn_model(input_shape=(150, 150, 3), num_classes=NUM_CLASSES)
    log_message("CNN model created successfully with augmentation and noise layers")
    
    print("\nModel Architecture:")
    model.summary()
    
    # Step 4: Train Model with Data Augmentation
    print("\n" + "=" * 70)
    log_message("STEP 4: Training model with data augmentation...")
    print("=" * 70)
    
    log_message(f"Training for {EPOCHS} epochs with class weight balancing")
    
    history = train_model_with_generators(
        model,
        train_generator,
        test_generator,
        experiment_dir,
        epochs=EPOCHS,
        class_weight_dict=class_weight_dict
    )
    
    log_message("Training completed!")
    
    # Step 5: Save Training History
    print("\n" + "=" * 70)
    log_message("STEP 5: Saving training history...")
    print("=" * 70)
    
    save_training_history(history, experiment_dir)
    log_message(f"Training history saved to {experiment_dir / 'history.json'}")
    
    # Step 6: Evaluate Model
    print("\n" + "=" * 70)
    log_message("STEP 6: Evaluating model...")
    print("=" * 70)
    
    log_message("Generating predictions on test set...")
    y_pred_proba = model.predict(test_generator)
    y_pred = y_pred_proba.argmax(axis=1)
    y_test = test_generator.classes
    
    log_message("Generating evaluation report...")
    generate_evaluation_report(
        y_test, y_pred, y_pred_proba,
        history,
        experiment_dir
    )
    
    # Step 7: Mislabel Audit
    print("\n" + "=" * 70)
    log_message("STEP 7: Running mislabel audit...")
    print("=" * 70)
    
    suspicious_df, low_conf_df = save_mislabel_report(
        y_test, y_pred, y_pred_proba,
        experiment_dir
    )
    
    # Step 8: Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    
    print(f"\n📊 Results Summary:")
    print(f"   Experiment directory: {experiment_dir}")
    print(f"   Training batches: {len(train_generator)}")
    print(f"   Test batches: {len(test_generator)}")
    print(f"   Number of classes: {NUM_CLASSES}")
    print(f"   Classes: {list(test_generator.class_indices.keys())}")
    
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
