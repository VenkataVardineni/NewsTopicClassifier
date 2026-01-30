"""
Flask API server for news topic classification.
Supports Logistic Regression, Linear SVM, XGBoost, and DistilBERT.
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

from src.features.build_features import clean_text
from src.features.build_features_nlp import clean_text as clean_text_nlp

app = Flask(__name__)
CORS(app)

# Global variables for models
models_cache = {}
vectorizer = None
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
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                sentence_model.save(model_path)
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
    return lda_model, lda_vectorizer

def get_nlp_features(text):
    """Extract NLP features using sentence transformer and optionally LDA."""
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
        features = np.hstack([embedding, topic_features])
    else:
        features = embedding
    
    return features.reshape(1, -1)

def load_tfidf_model(model_name):
    """Load TF-IDF based model (LR, SVM, XGBoost)."""
    global vectorizer, category_names
    
    model_path = f'models/{model_name}.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if vectorizer is None:
        vectorizer_path = 'models/vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    
    if category_names is None:
        category_names_path = 'models/category_names.pkl'
        with open(category_names_path, 'rb') as f:
            category_names = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model, vectorizer, category_names, 'tfidf'

def load_nlp_model(model_name):
    """Load NLP-based model (trained on semantic embeddings)."""
    global category_names
    
    model_path = f'models/{model_name}.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if category_names is None:
        category_names_path = 'models/category_names.pkl'
        with open(category_names_path, 'rb') as f:
            category_names = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model, category_names, 'nlp'

def load_transformer_model():
    """Load DistilBERT transformer model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_dir = 'models/distilbert'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"DistilBERT model not found at {model_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        with open('models/category_names.pkl', 'rb') as f:
            category_names = pickle.load(f)
        
        return model, tokenizer, category_names
    except ImportError:
        raise ImportError("transformers library not installed. Install with: pip install transformers torch")

def get_model(model_name):
    """Get model from cache or load it."""
    if model_name not in models_cache:
        if model_name == 'distilbert':
            models_cache[model_name] = load_transformer_model()
        elif '_nlp' in model_name:
            models_cache[model_name] = load_nlp_model(model_name)
        else:
            models_cache[model_name] = load_tfidf_model(model_name)
    return models_cache[model_name]

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
        
        # Clean text
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) == 0:
            return jsonify({'error': 'Text is empty after cleaning'}), 400
        
        # Get model
        model_result = get_model(requested_model)
        
        if requested_model == 'distilbert':
            # Transformer prediction
            model, tokenizer, category_names = model_result
            import torch
            
            encoding = tokenizer(
                cleaned_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
                prediction = int(np.argmax(probabilities))
        elif '_nlp' in requested_model:
            # NLP-based models (semantic embeddings)
            model, category_names, model_type = model_result
            X = get_nlp_features(text)
            if X is None:
                return jsonify({'error': 'Text is empty after cleaning'}), 400
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        elif requested_model == 'xgboost' or requested_model == 'xgboost_nlp':
            # XGBoost - check if NLP version exists
            xgboost_nlp_path = 'models/xgboost_nlp.pkl'
            if os.path.exists(xgboost_nlp_path):
                # Use NLP version
                model, category_names, model_type = load_nlp_model('xgboost_nlp')
                X = get_nlp_features(text)
                if X is None:
                    return jsonify({'error': 'Text is empty after cleaning'}), 400
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
            else:
                # XGBoost NLP model not available
                return jsonify({
                    'error': 'XGBoost NLP model not found. Please train it first or use another model like logistic_regression_nlp or linear_svm_nlp.'
                }), 404
        else:
            # TF-IDF based models
            model, vectorizer_obj, category_names, model_type = model_result
            X = vectorizer_obj.transform([cleaned_text])
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        
        # Get category name
        category = category_names[prediction]
        
        # Get top predictions
        top_predictions = []
        if probabilities is not None:
            if isinstance(probabilities, np.ndarray):
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
        if category_names is None:
            category_names_path = 'models/category_names.pkl'
            with open(category_names_path, 'rb') as f:
                cat_names = pickle.load(f)
        else:
            cat_names = category_names
        
        if hasattr(cat_names, 'tolist'):
            return jsonify({'categories': cat_names.tolist()})
        return jsonify({'categories': list(cat_names)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    available_models = []
    
    # Check for NLP models (preferred - these are the efficient ones)
    nlp_models = ['logistic_regression_nlp', 'linear_svm_nlp', 'random_forest_nlp']
    for model in nlp_models:
        if os.path.exists(f'models/{model}.pkl'):
            available_models.append(model)
    
    # Check for XGBoost NLP model
    if os.path.exists('models/xgboost_nlp.pkl'):
        available_models.append('xgboost_nlp')
    
    # Check for old TF-IDF models (fallback, but may have issues)
    old_models = ['logistic_regression', 'linear_svm']
    for model in old_models:
        if os.path.exists(f'models/{model}.pkl') and f'{model}_nlp' not in available_models:
            # Only add if NLP version doesn't exist
            available_models.append(model)
    
    # Check for DistilBERT
    if os.path.exists('models/distilbert'):
        available_models.append('distilbert')
    
    # If no models found, return at least the default NLP models
    if not available_models:
        available_models = ['logistic_regression_nlp', 'linear_svm_nlp']
    
    return jsonify({'models': available_models})

if __name__ == '__main__':
    # Try to load default model on startup
    try:
        get_model('logistic_regression')
        print("Default model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
        print("Please train the model first by running: python src/models/train.py")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
