# 🍎 Fruit Classifier - ML with Transfer Learning

An end-to-end fruit classification project using **MobileNetV2 transfer learning** for high accuracy with limited training data (240 images).

**Key Results:**
- ✅ **Test Accuracy: 91.67%** (55/60 images correct) - Significant improvement!
- ✅ **Per-Class**: Apple 95%, Banana 100%, Orange 100%, Mixed 20%
- ✅ **Architecture**: MobileNetV2 transfer learning with fine-tuned head
- ✅ **Data**: 193 training (80%), 47 validation (20%), 60 test (separate)
- ✅ **Improvements**: Class balancing, data augmentation, mislabel detection
- ✅ **Deployment**: Docker containerization + REST API + Standalone inference script

## 📚 Documentation

Comprehensive documentation files are located in `src/`:
- **[README.md](src/README.md)** - Full project documentation
- **[FINAL_RESULTS.md](src/FINAL_RESULTS.md)** - Complete results and requirement verification
- **[QUICK_REFERENCE.md](src/QUICK_REFERENCE.md)** - Quick lookup guide
- **[CLEANUP_REPORT.md](src/CLEANUP_REPORT.md)** - Project structure and cleanup details

## Project Structure

```
TeamX/
├─ src/                              # Source code and documentation
│  ├─ Image_Classifier_Training.ipynb # Main training notebook (RUN THIS!)
│  ├─ README.md                      # Full project documentation
│  ├─ FINAL_RESULTS.md               # Complete results summary
│  ├─ QUICK_REFERENCE.md             # Quick reference guide
│  ├─ CLEANUP_REPORT.md              # Project structure details
│  ├─ model.py                       # Model definitions (CNN + MobileNetV2)
│  ├─ config.py                      # Configuration and hyperparameters
│  ├─ data.py                        # Data loading and preprocessing
│  ├─ train.py                       # Training logic and callbacks
│  ├─ evaluate.py                    # Evaluation metrics
│  └─ utils.py                       # Utility functions
├─ Docker/                           # Containerization
│  ├─ Dockerfile                     # Docker image definition
│  ├─ docker-compose.yml             # Multi-service orchestration
│  ├─ api.py                         # REST API server
├─ data/                             # Dataset
│  ├─ train/                         # Training images organized by class
│  │  ├─ apple/
│  │  ├─ banana/
│  │  ├─ mixed/
│  │  └─ orange/
│  └─ test/                          # Test images (60 separate images)
├─ experiments/                      # Training results (auto-generated)
│  ├─ exp_002_baseline/              # Timestamped experiment folder
│  │  ├─ model_best.h5               # Best trained model
│  │  ├─ history.json                # Training history
│  │  ├── mislabels/                 # Suspicious prediction analysis
│  │  └── plots/                     # Training visualizations
├─ pdf/                              # Exported notebook PDFs
├─ standalone.py                     # Inference script (NO CONFIG DEPENDENCIES!)
├─ requirements.txt                  # Python dependencies
└─ README.md                         # This file (overview)
```

## Quick Start

### 1️⃣ Training (Jupyter Notebook)

```bash
cd TeamX
pip install -r requirements.txt
# Open src/Image_Classifier_Training.ipynb in VS Code and run all cells
```

The notebook will:
- ✅ Load and organize data
- ✅ Build MobileNetV2 transfer learning model
- ✅ Train with data augmentation
- ✅ Save best model to `experiments/`
- ✅ Generate evaluation metrics and plots

### 2️⃣ Inference (Standalone Script)

After training, use the independent prediction script in the root:

```bash
# Single image prediction (auto-detects latest model)
python standalone.py data/test/apple_77.jpg

# Batch processing (flat directory)
python standalone.py --batch ./data/test/

# Custom model path
python standalone.py image.jpg --model experiments/exp_002_baseline/model_best.h5
```

**Output files** (saved to `experiments/` with timestamps):
- Single: `prediction_20251130_125247.json`
- Batch: `batch_predictions_20251130_125247.json` + `.csv`

**Key Feature**: `standalone.py` is **completely independent** - no config files or project structure needed! Just copy it with your trained model anywhere.

### 3️⃣ Docker Deployment

Docker files are organized in the `Docker/` folder:

```bash
# Build image
docker build -f Docker/Dockerfile -t fruit-classifier .

# Run training
docker-compose -f Docker/docker-compose.yml up trainer

# REST API
docker-compose -f Docker/docker-compose.yml up api
# Then: curl -X POST -F "image=@photo.jpg" http://localhost:5000/predict

# Batch prediction
docker-compose -f Docker/docker-compose.yml up predictor
```

## Model Architecture

### MobileNetV2 Transfer Learning

Why transfer learning for this project?
- **Limited data**: Only 240 training images (too small for training CNN from scratch)
- **Pre-trained backbone**: MobileNetV2 trained on 1.3M ImageNet images
- **Better accuracy**: 31% → 60-70% test accuracy
- **Lightweight**: Only fine-tune top 2-3 layers

**Architecture:**
```
Input (150×150×3)
    ↓
MobileNetV2 (pre-trained on 1.4M ImageNet images, frozen base)
    ↓
GlobalAveragePooling2D()
    ↓
Dropout(0.5) → Dense(256, relu) → BatchNormalization → Dropout(0.3)
    ↓
Dense(4, softmax) → [apple, banana, mixed, orange]
```

**Training Details:**
- Learning Rate: 0.001 (conservative for transfer learning)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 16
- Epochs: 50 (early stopped at ~20 epochs)
- Early Stopping: Monitor val_accuracy, patience=6
- Data Augmentation: Rotation ±40°, Shift ±20%, Zoom ±20%, Brightness 0.8-1.2x, Horizontal flip
- Class Weights: Balanced to handle "mixed" class imbalance
- Data Split: 80% training (193 images), 20% validation (47 images), 60 test (separate)

## Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **91.67%** (55/60 images) |
| Apple Accuracy | 95% (18/19) |
| Banana Accuracy | 100% (18/18) |
| Orange Accuracy | 100% (18/18) |
| Mixed Accuracy | 20% (1/5) - small sample size |
| Macro Avg (Unweighted) | 79% |
| Weighted Avg | 90% |
| Model Size | 10.2 MB |
| Training Time | ~15-20 min (CPU) |
| Inference Time | ~50-100ms per image |

**Key Findings:**
- ✅ Excellent performance on well-represented classes (apple, banana, orange)
- ⚠️ Mixed class struggles due to only 5 test samples (needs more data)
- ✅ Transfer learning significantly improved accuracy
- ✅ Data augmentation and class balancing boosted generalization

## Validation & Testing

Run the validation script to verify all 4 execution paths are ready:

```bash
# Quick validation (1 minute)
python quick_validation.py

# Comprehensive testing (includes predictions)
python comprehensive_test.py
```

**Validation Results**: ✅ All 4 paths passing
1. ✅ File Organization - All required files present
2. ✅ Data Availability - 480 training + 120 test images
3. ✅ Model Configuration - MobileNetV2 + Early Stopping
4. ✅ Output Logging - Timestamped outputs to experiments/

## Files Reference

| File | Purpose |
|------|---------|
| `Image_Classifier_Training.ipynb` | Main training notebook with MobileNetV2 |
| `standalone.py` | Standalone inference (outputs to experiments/ with timestamps) |
| `quick_validation.py` | Fast validation of all 4 execution paths |
| `comprehensive_test.py` | Extended test suite with predictions |
| `model.py` | Model definitions (MobileNetV2 transfer learning) |
| `config.py` | Hyperparameters and paths |
| `data.py` | Data loading utilities |
| `train.py` | Training loops and callbacks |
| `Docker/Dockerfile` | Container image definition |
| `Docker/docker-compose.yml` | Multi-service orchestration |
| `Docker/api.py` | Flask REST API server |
