"""
Mislabel audit module for identifying potentially mislabeled training samples.
Analyzes model predictions to flag suspicious labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import MISLABEL_THRESHOLD, CLASS_NAMES, EXPERIMENTS_DIR


def identify_potential_mislabels(y_true, y_pred, y_pred_proba,
                                    true_class_proba_threshold=MISLABEL_THRESHOLD):
    """
    Identify potentially mislabeled samples.
    Flags samples where the model's top prediction probability for the true class
    is below the threshold, suggesting the label might be incorrect.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        true_class_proba_threshold: Threshold for flagging potential mislabels
        
    Returns:
        DataFrame with suspicious samples
    """
    true_class_proba = np.max(y_pred_proba, axis=1)
    
    # Get probability assigned to the true class
    true_class_confidence = y_pred_proba[np.arange(len(y_true)), y_true]
    
    # Flag samples where model predicts differently with high confidence
    suspicious_mask = (
        (y_pred != y_true) &
        (y_pred_proba[np.arange(len(y_true)), y_pred] > true_class_proba_threshold)
    )
    
    suspicious_indices = np.where(suspicious_mask)[0]
    
    mislabel_data = {
        'sample_index': suspicious_indices,
        'true_label': y_true[suspicious_indices],
        'predicted_label': y_pred[suspicious_indices],
        'confidence_true_class': true_class_confidence[suspicious_indices],
        'confidence_predicted_class': y_pred_proba[suspicious_indices, y_pred[suspicious_indices]],
        'confidence_diff': (y_pred_proba[suspicious_indices, y_pred[suspicious_indices]] -
                            true_class_confidence[suspicious_indices])
    }
    
    df = pd.DataFrame(mislabel_data)
    df = df.sort_values('confidence_diff', ascending=False)
    
    return df


def identify_low_confidence_predictions(y_true, y_pred, y_pred_proba,
                                        confidence_threshold=0.5):
    """
    Identify predictions with low confidence.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        confidence_threshold: Confidence threshold
        
    Returns:
        DataFrame with low confidence predictions
    """
    max_proba = np.max(y_pred_proba, axis=1)
    low_confidence_mask = max_proba < confidence_threshold
    low_confidence_indices = np.where(low_confidence_mask)[0]
    
    low_conf_data = {
        'sample_index': low_confidence_indices,
        'true_label': y_true[low_confidence_indices],
        'predicted_label': y_pred[low_confidence_indices],
        'confidence': max_proba[low_confidence_indices]
    }
    
    df = pd.DataFrame(low_conf_data)
    df = df.sort_values('confidence', ascending=True)
    
    return df


def save_mislabel_report(y_true, y_pred, y_pred_proba, experiment_dir,
                        class_names=CLASS_NAMES):
    """
    Generate and save mislabel audit report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        experiment_dir: Directory to save report
        class_names: List of class names
    """
    experiment_dir = Path(experiment_dir)
    mislabels_dir = experiment_dir / 'mislabels'
    mislabels_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify potential mislabels
    suspicious_df = identify_potential_mislabels(y_true, y_pred, y_pred_proba)
    
    # Map numeric labels to class names
    if len(suspicious_df) > 0:
        suspicious_df['true_class'] = suspicious_df['true_label'].map(
            {i: name for i, name in enumerate(class_names)}
        )
        suspicious_df['predicted_class'] = suspicious_df['predicted_label'].map(
            {i: name for i, name in enumerate(class_names)}
        )
    
    # Save to CSV
    suspicious_df.to_csv(mislabels_dir / 'suspected_mislabels.csv', index=False)
    
    # Also identify low confidence
    low_conf_df = identify_low_confidence_predictions(y_true, y_pred, y_pred_proba)
    if len(low_conf_df) > 0:
        low_conf_df['true_class'] = low_conf_df['true_label'].map(
            {i: name for i, name in enumerate(class_names)}
        )
        low_conf_df['predicted_class'] = low_conf_df['predicted_label'].map(
            {i: name for i, name in enumerate(class_names)}
        )
    
    low_conf_df.to_csv(mislabels_dir / 'low_confidence_predictions.csv', index=False)
    
    print(f"Mislabel audit report saved to {mislabels_dir}")
    print(f"Found {len(suspicious_df)} suspicious labels")
    print(f"Found {len(low_conf_df)} low confidence predictions")
    
    return suspicious_df, low_conf_df
