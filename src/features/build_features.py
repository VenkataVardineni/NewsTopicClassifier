"""
Feature engineering script for news topic classification.
Downloads dataset, cleans text, and creates TF-IDF feature matrices.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_features():
    """Download dataset, clean text, and create TF-IDF features."""
    
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
    # Clean texts
    cleaned_texts = [clean_text(text) for text in all_texts]
    
    # Filter out empty texts
    non_empty = [(text, label) for text, label in zip(cleaned_texts, all_labels) if len(text) > 0]
    cleaned_texts, all_labels = zip(*non_empty) if non_empty else ([], [])
    
    print(f"After cleaning: {len(cleaned_texts)} samples")
    
    print("Creating TF-IDF features...")
    # Create improved TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=30000,  # Balanced: enough features for good performance, not too memory-intensive
        min_df=2,  # Minimum document frequency
        max_df=0.85,  # Remove very common words
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams for better context
        stop_words='english',
        sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf)) - reduces impact of high frequency terms
        norm='l2',  # L2 normalization
        analyzer='word'  # Word-level analysis
    )
    
    # Fit and transform
    X = vectorizer.fit_transform(cleaned_texts)
    y = np.array(all_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save feature matrices and labels
    print("Saving features...")
    with open('data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/X_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)
    with open('data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save category names
    with open('models/category_names.pkl', 'wb') as f:
        pickle.dump(newsgroups_train.target_names, f)
    
    # Save metadata
    categories_list = newsgroups_train.target_names
    if hasattr(categories_list, 'tolist'):
        categories_list = categories_list.tolist()
    elif not isinstance(categories_list, list):
        categories_list = list(categories_list)
    
    metadata = {
        'n_samples': len(cleaned_texts),
        'n_features': X.shape[1],
        'n_categories': len(newsgroups_train.target_names),
        'categories': categories_list
    }
    
    with open('data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Feature engineering complete!")
    print(f"Categories: {', '.join(newsgroups_train.target_names[:10])}...")
    
    return X_train, X_val, y_train, y_val, vectorizer

if __name__ == '__main__':
    build_features()

