"""
CLI tool for predicting news article topics.
Supports Logistic Regression, Linear SVM, XGBoost, and DistilBERT.
"""

import sys
import pickle
import argparse
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features import clean_text

def load_model_and_vectorizer(model_name='logistic_regression'):
    """Load trained model and vectorizer."""
    # Check if it's a transformer model
    if model_name == 'distilbert':
        return load_transformer_model()
    
    model_path = f'models/{model_name}.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    category_names_path = 'models/category_names.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python src/models/train.py")
        sys.exit(1)
    
    if not os.path.exists(vectorizer_path):
        print(f"Error: Vectorizer file not found at {vectorizer_path}")
        print("Please build features first by running: python src/features/build_features.py")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(category_names_path, 'rb') as f:
        category_names = pickle.load(f)
    
    return model, vectorizer, category_names, 'tfidf'

def load_transformer_model():
    """Load DistilBERT transformer model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_dir = 'models/distilbert'
        if not os.path.exists(model_dir):
            print(f"Error: DistilBERT model not found at {model_dir}")
            print("Please train the model first by running: python src/models/train_transformer.py")
            sys.exit(1)
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        with open('models/category_names.pkl', 'rb') as f:
            category_names = pickle.load(f)
        
        return model, tokenizer, category_names, 'transformer'
    except ImportError:
        print("Error: transformers library not installed.")
        print("Install it with: pip install transformers torch")
        sys.exit(1)

def predict(text, model_name='logistic_regression'):
    """Predict topic for given text."""
    # Load model and vectorizer
    result = load_model_and_vectorizer(model_name)
    
    if len(result) == 4:
        model, vectorizer_or_tokenizer, category_names, model_type = result
    else:
        # Fallback for old format
        model, vectorizer_or_tokenizer, category_names = result
        model_type = 'tfidf'
    
    # Clean text
    cleaned_text = clean_text(text)
    
    if len(cleaned_text) == 0:
        print("Error: Text is empty after cleaning.")
        return None
    
    # Predict based on model type
    if model_type == 'transformer':
        # Transformer prediction
        import torch
        encoding = vectorizer_or_tokenizer(
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
    else:
        # TF-IDF based models
        X = vectorizer_or_tokenizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    
    # Get category name
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
        description='Predict news article topic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cli/predict.py "The Lakers won the championship last night"
  python src/cli/predict.py "Apple announced new iPhone features" --model xgboost
  python src/cli/predict.py "Tech news article" --model distilbert
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='News article text to classify'
    )
    
    parser.add_argument(
        '--model',
        default='logistic_regression',
        choices=['logistic_regression', 'linear_svm', 'xgboost', 'distilbert'],
        help='Model to use for prediction (default: logistic_regression)'
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
