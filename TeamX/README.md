# Fruit Classifier - Machine Learning Project

A machine learning project for classifying different types of fruits using deep learning and transfer learning approaches.

## Project Structure

```
TeamX/
├─ src/                      # Source code modules
│  ├─ config.py              # Configuration and hyperparameters
│  ├─ data.py                # Data loading and preprocessing
│  ├─ model.py               # Model architecture definitions
│  ├─ train.py               # Training logic and callbacks
│  ├─ evaluate.py            # Evaluation metrics and visualization
│  ├─ utils.py               # Utility functions
│  └─ mislabel_audit.py      # Mislabel detection and analysis
├─ data/                     # Dataset
│  ├─ train/                 # Training images
│  └─ test/                  # Test images
├─ experiments/              # Training results (auto-generated)
│  └─ exp_XXX_baseline/      # Each experiment creates a new folder
│     ├─ history.json        # Training history
│     ├─ metrics.json        # Evaluation metrics
│     ├─ model_best.h5       # Best model checkpoint
│     ├─ plots/              # Generated visualizations
│     └─ mislabels/          # Mislabel audit reports
├─ checklist/                # Verification scripts and docs
│  ├─ verify_data.py         # Data verification script
│  ├─ test_imports.py        # Import testing script
│  └─ *.md                   # Documentation files
├─ main.py                   # Main training script (RUN THIS!)
├─ requirements.txt          # Python dependencies
└─ README.md                 # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup** (optional):
   ```bash
   python checklist/verify_data.py
   python checklist/test_imports.py
   ```

3. **Data & Configuration**:
   - ✅ Data is already set up in `data/train/` and `data/test/`
   - ✅ Configuration is set for 4 classes: apple, banana, mixed, orange
   - Edit `src/config.py` to adjust hyperparameters if needed

## Path Configuration

All paths are **relative**, making the project portable across machines without modification:

- **config.py** - Central hub managing all paths
  - Uses `Path(__file__).parent.parent` for module imports
  - Falls back to `Path.cwd()` for Jupyter notebooks
- **All data references** go through `config.py` and `data.py`
- **No hardcoded paths** - Works on any machine, any OS

### Running from Different Contexts

**Python script from TeamX/ root:**

```bash
python main.py
```

**Jupyter notebook from TeamX/src/:**

- Open `Image_Classifier_Training.ipynb` in VS Code
- Notebook automatically finds `data/` in relative path

**After cloning from GitHub:**

```bash
git clone <repo-url>
cd ML-CA-Image-Classifier-V1.0/TeamX
python main.py  # Works immediately, no path config needed!
```

## Usage

### Quick Start - Run Complete Pipeline

The easiest way to train your model:

```bash
python main.py
```

This will automatically:

1. Load training and test data
2. Preprocess images
3. Create and train the CNN model
4. Generate evaluation metrics and plots
5. Run mislabel audit
6. Save all results to `experiments/` directory

### Experiment Outputs

When you run the training pipeline (either via `main.py` or the Jupyter notebook), results are saved to:

```bash
experiments/
└─ notebook_YYYYMMDD_HHMMSS/   # Timestamped experiment folder
   ├─ model_best.h5             # Best trained model
   ├─ history.json              # Training/validation metrics per epoch
   ├─ metrics.json              # Final accuracy & class distribution
   ├─ training_history.png      # Accuracy & loss plots
   └─ confusion_matrix.png      # Confusion matrix visualization
```

**Note:** The `experiments/` folder is **not tracked by Git** (.gitignore excludes it). This keeps the repository lightweight. Generate experiments locally by running the training pipeline.

### Manual Training (Advanced)

If you want more control, you can use the modules directly:

```python
import sys
sys.path.insert(0, 'src')

from data import load_images_from_flat_directory, preprocess_images
from model import create_cnn_model
from train import train_model
from evaluate import generate_evaluation_report
from config import DATA_DIR
from tensorflow.keras.utils import to_categorical

# Load data
X_train, y_train, classes = load_images_from_flat_directory(DATA_DIR / "train")
X_test, y_test, _ = load_images_from_flat_directory(DATA_DIR / "test")

# Preprocess
X_train = preprocess_images(X_train)
X_test = preprocess_images(X_test)

# Convert labels
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Train
model = create_cnn_model(num_classes=4)
history = train_model(model, X_train, y_train_cat, X_test, y_test_cat, 'experiments/exp_001/')

# Evaluate
y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)
generate_evaluation_report(y_test, y_pred, y_pred_proba, history, 'experiments/exp_001/')
```

## Features

- **Multiple Model Architectures**: CNN and Transfer Learning (MobileNetV2)
- **Data Augmentation**: Automatic augmentation during training
- **Comprehensive Evaluation**: Metrics, confusion matrix, ROC curves
- **Mislabel Detection**: Identifies potentially mislabeled training samples
- **Experiment Tracking**: Organized experiment results with versioning
- **Easy Configuration**: Centralized hyperparameter management

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- scikit-learn 1.0+
- pandas, numpy, matplotlib

## Notes

- Ensure images are in standard formats (JPEG, PNG)
- Recommended image size: 224x224 pixels
- Adjust `BATCH_SIZE` and `EPOCHS` based on your hardware
- Use GPU for faster training (requires CUDA/cuDNN)

## Authors

NUS-ISS Team Project - SA4110 Machine Learning Application Development
