# Quick Reference - Fruit Classifier

## 📊 Your Dataset

- **Classes**: apple, banana, mixed, orange (4 classes)
- **Training**: 240 images (75 apple, 73 banana, 20 mixed, 72 orange)
- **Testing**: 60 images (19 apple, 18 banana, 5 mixed, 18 orange)
- **Image Size**: 224x224 pixels

## 🚀 Quick Start Commands

### Verify Setup
```bash
python verify_data.py
```

### Test Imports
```bash
python test_imports.py
```

## 📝 Basic Training Script

```python
import sys
sys.path.insert(0, 'src')

from data import load_images_from_flat_directory, preprocess_images
from model import create_cnn_model
from train import train_model, save_training_history
from evaluate import generate_evaluation_report
from config import DATA_DIR
from tensorflow.keras.utils import to_categorical

# Load data
print("Loading training data...")
X_train, y_train, classes = load_images_from_flat_directory(DATA_DIR / "train")
X_train = preprocess_images(X_train)

print("Loading test data...")
X_test, y_test, _ = load_images_from_flat_directory(DATA_DIR / "test")
X_test = preprocess_images(X_test)

# Convert to one-hot
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Create model
print("Creating model...")
model = create_cnn_model(num_classes=4)
model.summary()

# Train
print("Training model...")
history = train_model(
    model, 
    X_train, y_train_cat,
    X_test, y_test_cat,
    'experiments/exp_001_baseline/',
    epochs=50
)

# Evaluate
print("Evaluating model...")
y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

generate_evaluation_report(
    y_test, y_pred, y_pred_proba, 
    history, 
    'experiments/exp_001_baseline/'
)

print("Done! Check experiments/exp_001_baseline/ for results")
```

## 📁 Key Files

- `src/config.py` - Configuration (classes, hyperparameters)
- `src/data.py` - Data loading functions
- `src/model.py` - Model architectures
- `src/train.py` - Training logic
- `src/evaluate.py` - Evaluation and metrics
- `verify_data.py` - Data verification script

## ⚙️ Configuration Highlights

```python
NUM_CLASSES = 4
CLASS_NAMES = ["apple", "banana", "mixed", "orange"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 🎯 Next Steps

1. Run `verify_data.py` to confirm setup
2. Create a training script (or use the example above)
3. Monitor training in `experiments/` directory
4. Review metrics and plots after training
5. Iterate on model architecture or hyperparameters

## 💡 Tips

- **Class Imbalance**: "mixed" class has fewer samples - consider class weights
- **GPU**: Training will be faster with GPU (CUDA/cuDNN)
- **Experiments**: Each run creates a new experiment directory
- **Checkpoints**: Best model is saved automatically during training

## 🔧 Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**Data not loading?**
- Check that files are in `data/train/` and `data/test/`
- Verify filenames follow pattern: `classname_number.jpg`

**Out of memory?**
- Reduce `BATCH_SIZE` in `src/config.py`
- Use smaller `IMAGE_SIZE`

---

All set! Your project is configured and ready to train. 🎉
