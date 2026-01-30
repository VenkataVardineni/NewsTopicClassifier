"""
Flask API server for news topic classification using NLP embeddings.
Uses sentence transformers for semantic understanding.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features_nlp import clean_text

app = Flask(__name__)
CORS(app)

# Global variables
models_cache = {}
sentence_model = None
lda_model = None
lda_vectorizer = None
category_names = None

def load_sentence_transformer():
    """Load sentence transformer model."""
    global sentence_model
    if sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_path = 'models/sentence_transformer'
            if os.path.exists(model_path):
                sentence_model = SentenceTransformer(model_path)
            else:
                # Load pre-trained model
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                sentence_model.save(model_path)
            print("Sentence transformer loaded")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    return sentence_model

def load_lda_models():
    """Load LDA models if available."""
    global lda_model, lda_vectorizer
    if lda_model is None and os.path.exists('models/lda_model.pkl'):
        with open('models/lda_model.pkl', 'rb') as f:
            lda_model = pickle.load(f)
        with open('models/lda_vectorizer.pkl', 'rb') as f:
            lda_vectorizer = pickle.load(f)
        print("LDA models loaded")
    return lda_model, lda_vectorizer

def get_features(text):
    """Extract features using sentence transformer and optionally LDA."""
    # Clean text
    cleaned_text = clean_text(text)
    
    if len(cleaned_text) == 0:
        return None
    
    # Get sentence embedding
    sentence_model = load_sentence_transformer()
    embedding = sentence_model.encode([cleaned_text])[0]
    
    # Get LDA topic features if available
    lda_model, lda_vectorizer = load_lda_models()
    if lda_model is not None and lda_vectorizer is not None:
        X_counts = lda_vectorizer.transform([cleaned_text])
        topic_features = lda_model.transform(X_counts)[0]
        # Combine features
        features = np.hstack([embedding, topic_features])
    else:
        features = embedding
    
    return features.reshape(1, -1)

def load_model(model_name):
    """Load trained classifier model."""
    if model_name not in models_cache:
        model_path = f'models/{model_name}.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            models_cache[model_name] = pickle.load(f)
        
        # Load category names
        global category_names
        if category_names is None:
            with open('models/category_names.pkl', 'rb') as f:
                category_names = pickle.load(f)
    
    return models_cache[model_name], category_names

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict topic for given text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        requested_model = data.get('model', 'logistic_regression_nlp')
        
        # Get features
        X = get_features(text)
        if X is None:
            return jsonify({'error': 'Text is empty after cleaning'}), 400
        
        # Load model
        model, category_names = load_model(requested_model)
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
        
        # Get category name
        category = category_names[prediction]
        
        # Get top predictions
        top_predictions = []
        if probabilities is not None:
            top_indices = sorted(
                range(len(probabilities)),
                key=lambda i: probabilities[i],
                reverse=True
            )[:5]
            top_predictions = [
                {
                    'category': category_names[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top_indices
            ]
        
        return jsonify({
            'category': category,
            'category_id': int(prediction),
            'top_predictions': top_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of all categories."""
    try:
        global category_names
        if category_names is None:
            with open('models/category_names.pkl', 'rb') as f:
                category_names = pickle.load(f)
        
        if hasattr(category_names, 'tolist'):
            return jsonify({'categories': category_names.tolist()})
        return jsonify({'categories': list(category_names)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    available_models = []
    
    # Check for NLP models
    nlp_models = ['logistic_regression_nlp', 'linear_svm_nlp', 'random_forest_nlp']
    for model in nlp_models:
        if os.path.exists(f'models/{model}.pkl'):
            available_models.append(model)
    
    # Fallback to old models if NLP models don't exist
    if not available_models:
        old_models = ['logistic_regression', 'linear_svm']
        for model in old_models:
            if os.path.exists(f'models/{model}.pkl'):
                available_models.append(model)
    
    return jsonify({'models': available_models})

if __name__ == '__main__':
    # Try to load models on startup
    try:
        load_sentence_transformer()
        print("NLP models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load NLP models: {e}")
        print("Please build features first: python src/features/build_features_nlp.py")
    
    app.run(host='0.0.0.0', port=5001, debug=True)

