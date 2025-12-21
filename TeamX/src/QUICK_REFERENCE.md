# Quick Reference - Fruit Classifier Results

## 🎯 Key Results at a Glance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **91.67%** |
| Apple Accuracy | 95% |
| Banana Accuracy | 100% |
| Orange Accuracy | 100% |
| Mixed Accuracy | 20% |

## 🏗️ Model Stack
- **Base**: MobileNetV2 (pre-trained on 1.4M ImageNet images)
- **Input**: 150×150×3 RGB images
- **Head**: GlobalAveragePooling2D → Dropout(0.5) → Dense(256) → BatchNorm → Dropout(0.3) → Dense(4)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy

## 📊 Data Split
- **Training**: 193 images (80% of 240 from Train.zip)
- **Validation**: 47 images (20% of 240) - used for monitoring during training
- **Test**: 60 images (from Test.zip) - completely separate, evaluated only after training

## 🔧 Improvements Applied
1. ✅ **Class Balancing** - Balanced weights for imbalanced classes
2. ✅ **Data Augmentation** - Rotation ±40°, Shift ±20%, Zoom ±20%, Brightness 0.8-1.2x
3. ✅ **Transfer Learning** - MobileNetV2 with frozen base + fine-tuned head
4. ✅ **Mislabel Detection** - Identifies suspicious/low-confidence predictions

## 📁 Output Files
```
experiments/notebook_YYYYMMDD_HHMMSS/
├── model_best.h5                    # Trained model
├── history.json                     # Training history per epoch
├── experiment_documentation.json    # Full experiment details
├── mislabel_report.json            # Suspicious image analysis
├── training_history.png            # Accuracy & loss curves
└── confusion_matrix.png            # Per-class predictions
```

## 🚀 Quick Commands

**Train the model:**
```bash
cd TeamX
python -m jupyter notebook src/Image_Classifier_Training.ipynb
# Run all cells
```

**After training, use standalone inference:**
```bash
# Single image
python standalone.py data/test/apple_77.jpg

# Batch processing
python standalone.py --batch ./data/test/
```

## 📈 Training Details
- **Epochs**: 50 (early stopped at ~20)
- **Batch Size**: 16
- **Learning Rate**: 0.001 (conservative for transfer learning)
- **Early Stopping**: Patience=6 epochs
- **Time to Train**: ~15-20 minutes (CPU)

## ⚠️ Known Limitations
- Mixed class has low accuracy (20%) due to only 5 test samples
- Would benefit from more "mixed" fruit training data
- Transfer learning works best with data similar to ImageNet (natural images)

## ✅ Project Requirements Status
1. ✅ CNN model for 4-class classification
2. ✅ Uses Train.zip (240) and Test.zip (60)
3. ✅ Documented experiments and results
4. ✅ Applied improvements (class balance, augmentation, transfer learning, mislabel detection)
5. ✅ Generated training history and confusion matrix plots
6. ✅ Comprehensive experiment documentation

---
**Test Accuracy: 91.67%** | **Status: COMPLETE** ✅
