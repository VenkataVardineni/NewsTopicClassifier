"""
Efficient NLP-based feature engineering using sentence transformers and topic modeling.
Uses semantic embeddings instead of heavy TF-IDF matrices.
"""

import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_features_nlp():
    """Build features using sentence transformers and topic modeling."""
    
    print("Downloading 20 Newsgroups dataset...")
    # Download dataset
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )
    
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )
    
    print(f"Training samples: {len(newsgroups_train.data)}")
    print(f"Test samples: {len(newsgroups_test.data)}")
    print(f"Categories: {len(newsgroups_train.target_names)}")
    
    # Combine train and test for full dataset
    all_texts = newsgroups_train.data + newsgroups_test.data
    all_labels = list(newsgroups_train.target) + list(newsgroups_test.target)
    
    print("Cleaning text...")
    # Clean texts (minimal cleaning for transformers)
    cleaned_texts = [clean_text(text) for text in all_texts]
    
    # Filter out empty texts
    non_empty = [(text, label) for text, label in zip(cleaned_texts, all_labels) if len(text) > 10]
    cleaned_texts, all_labels = zip(*non_empty) if non_empty else ([], [])
    
    print(f"After cleaning: {len(cleaned_texts)} samples")
    
    # Method 1: Sentence Transformers (semantic embeddings)
    print("\nCreating semantic embeddings with sentence transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use lightweight but effective model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast, good quality
        print("Model: all-MiniLM-L6-v2 (lightweight sentence transformer)")
        
        # Generate embeddings (384 dimensions - much smaller than 30k TF-IDF!)
        embeddings = model.encode(cleaned_texts, show_progress_bar=True, batch_size=32)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save sentence transformer model
        os.makedirs('models', exist_ok=True)
        model.save('models/sentence_transformer')
        print("Sentence transformer model saved")
        
    except ImportError:
        print("sentence-transformers not installed. Using fallback method...")
        embeddings = None
    
    # Method 2: Topic Modeling (LDA) for additional semantic features
    print("\nCreating topic modeling features...")
    try:
        # Use CountVectorizer for LDA (simpler than TF-IDF)
        vectorizer = CountVectorizer(
            max_features=5000,  # Much smaller than TF-IDF
            min_df=2,
            max_df=0.95,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X_counts = vectorizer.fit_transform(cleaned_texts)
        print(f"Count matrix shape: {X_counts.shape}")
        
        # LDA for topic modeling (20 topics for 20 categories)
        n_topics = 20
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,  # Fewer iterations for speed
            n_jobs=-1
        )
        
        print("Training LDA model...")
        topic_features = lda.fit_transform(X_counts)
        print(f"Topic features shape: {topic_features.shape}")
        
        # Save LDA model and vectorizer
        with open('models/lda_model.pkl', 'wb') as f:
            pickle.dump(lda, f)
        with open('models/lda_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
    except Exception as e:
        print(f"LDA failed: {e}")
        topic_features = None
    
    # Combine features
    print("\nCombining features...")
    if embeddings is not None and topic_features is not None:
        # Combine sentence embeddings + topic features
        X = np.hstack([embeddings, topic_features])
        print(f"Combined features shape: {X.shape}")
        print(f"Total features: {X.shape[1]} (vs 30,000 TF-IDF features)")
    elif embeddings is not None:
        X = embeddings
        print(f"Using sentence embeddings only: {X.shape}")
    elif topic_features is not None:
        X = topic_features
        print(f"Using topic features only: {X.shape}")
    else:
        raise ValueError("No features could be created!")
    
    y = np.array(all_labels)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save feature matrices and labels
    print("\nSaving features...")
    with open('data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/X_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)
    with open('data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    
    # Save category names
    categories_list = newsgroups_train.target_names
    if hasattr(categories_list, 'tolist'):
        categories_list = categories_list.tolist()
    elif not isinstance(categories_list, list):
        categories_list = list(categories_list)
    
    with open('models/category_names.pkl', 'wb') as f:
        pickle.dump(categories_list, f)
    
    # Save metadata
    metadata = {
        'n_samples': len(cleaned_texts),
        'n_features': X.shape[1],
        'n_categories': len(newsgroups_train.target_names),
        'categories': categories_list,
        'feature_type': 'sentence_transformer_lda' if (embeddings is not None and topic_features is not None) else ('sentence_transformer' if embeddings is not None else 'lda')
    }
    
    with open('data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nFeature engineering complete!")
    print(f"Categories: {', '.join(categories_list[:10])}...")
    print(f"\nFeature size: {X.shape[1]} dimensions (much smaller than TF-IDF!)")
    
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    build_features_nlp()

