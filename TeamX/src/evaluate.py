"""
Evaluation module for model performance assessment.
Includes metrics calculation, visualization, and reporting.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from config import CLASS_NAMES, EXPERIMENTS_DIR


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names=CLASS_NAMES, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        save_path: Path to save plot
    """
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def generate_evaluation_report(y_true, y_pred, y_pred_proba, history, experiment_dir):
    """
    Generate comprehensive evaluation report with plots and metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        history: Training history
        experiment_dir: Directory to save outputs
    """
    experiment_dir = Path(experiment_dir)
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, save_path=plots_dir / 'confusion_matrix.png')
    plot_training_history(history, save_path=plots_dir / 'loss_accuracy.png')
    plot_roc_curves(y_true, y_pred_proba, save_path=plots_dir / 'roc_curves.png')
    
    # Save metrics
    save_metrics(metrics, experiment_dir / 'metrics.json')
    
    print(f"Evaluation report saved to {experiment_dir}")
