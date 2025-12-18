# Team X - Machine Learning Image Classifier

A machine learning project for classifying fruit images (apples, bananas, oranges, and mixed combinations).

## Project Overview

This project implements a deep learning-based image classifier trained to recognize different types of fruits. It includes:
- Data preprocessing and augmentation
- Model training and evaluation
- Mislabel detection and auditing
- Performance metrics and visualization

## Prerequisites

- Python 3.8 or higher
- Git (with Git LFS support)
- pip or conda for package management

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shafikhakim27/Team-Project---Copy.git
cd Team\ Project\ -\ Copy
```

### 2. Install Git LFS (Important!)

This project uses **Git Large File Storage (Git LFS)** to manage large model files (>100MB).

**On Windows:**
```bash
# Using Chocolatey
choco install git-lfs

# Or download from
https://git-lfs.github.com/
```

**On macOS:**
```bash
brew install git-lfs
```

**On Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Fedora/RHEL
sudo yum install git-lfs
```

**Then initialize Git LFS:**
```bash
git lfs install
git lfs pull  # Pull large files from the repository
```

### 3. Install Python Dependencies

```bash
pip install -r TeamX/requirements.txt
```

## Project Structure

```
TeamX/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── run.bat                          # Batch script to run the application
├── src/
│   ├── config.py                   # Configuration settings
│   ├── data.py                     # Data loading and preprocessing
│   ├── model.py                    # Model architecture
│   ├── train.py                    # Training logic
│   ├── evaluate.py                 # Evaluation metrics
│   ├── mislabel_audit.py           # Mislabel detection
│   └── utils.py                    # Utility functions
├── data/
│   ├── train/                      # Training images
│   └── test/                       # Test images
├── experiments/
│   └── exp_002_baseline/           # Baseline model experiments
│       ├── model_best.h5           # Best trained model (stored with Git LFS)
│       ├── history.json            # Training history
│       ├── metrics.json            # Performance metrics
│       ├── mislabels/              # Mislabel analysis
│       └── plots/                  # Visualization plots
└── checklist/                      # Project documentation

```

## Usage

### Running the Classifier

```bash
# Navigate to TeamX directory
cd TeamX

# Run the application
python main.py

# Or use the batch script (Windows)
run.bat
```

### Training a New Model

```bash
python src/train.py
```

### Evaluating the Model

```bash
python src/evaluate.py
```

### Detecting Mislabeled Data

```bash
python src/mislabel_audit.py
```

## Model Information

- **Model Type**: Deep Neural Network
- **Framework**: TensorFlow/Keras
- **Input**: RGB Images (variable size)
- **Output**: Fruit classification (apple, banana, orange, mixed)
- **Best Model**: Stored as `experiments/exp_002_baseline/model_best.h5`

## Dataset

- **Training Set**: ~300 images
- **Test Set**: 60 images
- **Classes**: 4 (apple, banana, orange, mixed)
- **Image Format**: JPEG
- **Location**: `data/train/` and `data/test/`

## Troubleshooting

### Git LFS Issues

**If you see warnings about LFS:**
```bash
# Make sure Git LFS is installed
git lfs --version

# Reinitialize Git LFS
git lfs install --force
git lfs pull
```

**If the model file is not downloading:**
```bash
# Force download LFS objects
git lfs fetch --all
git lfs checkout
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install --upgrade -r TeamX/requirements.txt
```

## Project Status

- ✅ Data collection and preprocessing
- ✅ Model training and evaluation
- ✅ Mislabel detection
- ✅ Performance visualization
- 📊 Baseline accuracy: ~95%

## Team Members

This is a team project for NUS-ISS Machine Learning Application Development (SA4110)

## License

This project is for educational purposes.

## Support

For issues or questions, please refer to the project documentation in the `checklist/` folder.
