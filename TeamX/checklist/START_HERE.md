# 🚀 START HERE - Fruit Classifier Quick Start

## One Command to Rule Them All

```bash
python main.py
```

That's it! This will:
- ✅ Load your training data (240 images)
- ✅ Load your test data (60 images)
- ✅ Train a CNN model
- ✅ Evaluate performance
- ✅ Generate plots and metrics
- ✅ Run mislabel audit
- ✅ Save everything to `experiments/`

## First Time Setup

1. **Install dependencies** (one time only):
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup** (optional):
   ```bash
   python checklist/verify_data.py
   ```

3. **Run training**:
   ```bash
   python main.py
   ```

## What You'll Get

After running `main.py`, check the `experiments/exp_XXX_baseline/` folder for:

- 📊 **Confusion Matrix** - See which classes are confused
- 📈 **Training Curves** - Loss and accuracy over time
- 📉 **ROC Curves** - Performance per class
- 🎯 **Metrics** - Accuracy, F1-score, etc.
- 🔍 **Mislabel Report** - Potentially mislabeled samples
- 💾 **Trained Model** - Best model checkpoint

## Configuration

Want to change settings? Edit `src/config.py`:

```python
EPOCHS = 50              # Number of training epochs
BATCH_SIZE = 32          # Batch size
LEARNING_RATE = 0.001    # Learning rate
IMAGE_SIZE = (224, 224)  # Image dimensions
```

## Your Dataset

- **Classes**: apple, banana, mixed, orange (4 classes)
- **Training**: 240 images
- **Testing**: 60 images

## Need Help?

- 📖 Full documentation: `README.md`
- ✅ Verification scripts: `checklist/`
- 📝 Cleanup summary: `PROJECT_CLEANUP_SUMMARY.md`

## Tips

- Each run creates a new experiment folder (exp_001, exp_002, etc.)
- Training takes ~5-15 minutes depending on your hardware
- GPU will speed things up significantly
- Check the console output for progress

---

**Ready? Just run:** `python main.py` 🎉
