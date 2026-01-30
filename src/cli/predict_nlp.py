"""
CLI tool for predicting news article topics using NLP embeddings.
"""

import sys
import pickle
import argparse
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features_nlp import clean_text

def load_sentence_transformer():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        model_path = 'models/sentence_transformer'
        if os.path.exists(model_path):
            return SentenceTransformer(model_path)
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.save(model_path)
            return model
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install it with: pip install sentence-transformers")
        sys.exit(1)

def load_lda_models():
    """Load LDA models if available."""
    if os.path.exists('models/lda_model.pkl'):
        with open('models/lda_model.pkl', 'rb') as f:
            lda_model = pickle.load(f)
        with open('models/lda_vectorizer.pkl', 'rb') as f:
            lda_vectorizer = pickle.load(f)
        return lda_model, lda_vectorizer
    return None, None

def get_features(text):
    """Extract features using sentence transformer and optionally LDA."""
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

def predict(text, model_name='logistic_regression_nlp'):
    """Predict topic for given text."""
    # Get features
    X = get_features(text)
    if X is None:
        print("Error: Text is empty after cleaning.")
        return None
    
    # Load model
    model_path = f'models/{model_name}.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python src/models/train_nlp.py")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open('models/category_names.pkl', 'rb') as f:
        category_names = pickle.load(f)
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    
    category = category_names[prediction]
    
    return {
        'category': category,
        'category_id': int(prediction),
        'probabilities': probabilities,
        'category_names': category_names
    }

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Predict news article topic using NLP embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cli/predict_nlp.py "The Lakers won the championship last night"
  python src/cli/predict_nlp.py "Apple announced new iPhone features" --model random_forest_nlp
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='News article text to classify'
    )
    
    parser.add_argument(
        '--model',
        default='logistic_regression_nlp',
        choices=['logistic_regression_nlp', 'linear_svm_nlp', 'random_forest_nlp'],
        help='Model to use for prediction (default: logistic_regression_nlp)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Show top K predictions with probabilities (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Get text from argument or stdin
    if args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)
    
    # Predict
    result = predict(text, args.model)
    
    if result is None:
        sys.exit(1)
    
    # Print results
    print(f"\nPredicted Topic: {result['category']}")
    print(f"Category ID: {result['category_id']}")
    
    # Show top K predictions if probabilities available
    if result['probabilities'] is not None:
        print(f"\nTop {args.top_k} Predictions:")
        top_indices = result['probabilities'].argsort()[-args.top_k:][::-1]
        for idx in top_indices:
            prob = result['probabilities'][idx]
            print(f"  {result['category_names'][idx]:<30} {prob:.4f}")

if __name__ == '__main__':
    main()

