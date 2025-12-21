"""
Optional Flask REST API for serving predictions via HTTP
Usage: python Docker/api.py

Then POST images to http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import os

app = Flask(__name__)

# Global model cache
_model = None
_class_names = None


def load_model_cached():
    """Load model once on startup"""
    global _model, _class_names
    if _model is None:
        experiments_dir = Path('experiments')
        models = sorted(experiments_dir.glob('*/model_best.h5'))
        if not models:
            raise Exception("No saved model found!")
        _model = load_model(str(models[-1]))
        
        # Get class names
        data_dir = Path('data')
        if (data_dir / 'train' / 'organized').exists():
            _class_names = sorted([
                d.name for d in (data_dir / 'train' / 'organized').iterdir() 
                if d.is_dir()
            ])
        else:
            _class_names = ['apple', 'banana', 'mixed', 'orange']
    
    return _model, _class_names


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fruit type from uploaded image
    
    Request: POST /predict
    Body: multipart/form-data with 'image' file
    
    Response: JSON with prediction and confidence
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        model, class_names = load_model_cached()
        input_size = model.input_shape[1:3]
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            
            # Load and preprocess
            img = image.load_img(tmp.name, target_size=input_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            
            # Cleanup
            os.unlink(tmp.name)
            
            return jsonify({
                'prediction': class_names[pred_idx],
                'confidence': confidence,
                'all_probabilities': {
                    class_names[i]: float(predictions[0][i])
                    for i in range(len(class_names))
                }
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        model, _ = load_model_cached()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'input_shape': model.input_shape
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Loading model...")
    load_model_cached()
    print("✓ Model loaded. Starting API on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
