# News Topic Classifier

A machine learning application that classifies news articles into topics using efficient NLP models with semantic embeddings.

## Overview

This project provides a CLI and web interface to classify news articles into 20 categories (sports, tech, politics, business, etc.). It uses **semantic embeddings** (sentence transformers) and **topic modeling** instead of heavy TF-IDF matrices, making it:

- **75x smaller features**: 404 dimensions vs 30,000 TF-IDF features
- **Faster training**: Models train in seconds instead of minutes
- **Better accuracy**: Semantic understanding improves classification (~74% accuracy)
- **Lower disk usage**: Compact models and embeddings

## Features

- **Web Interface**: React-based frontend with real-time predictions
- **CLI Tool**: Command-line interface for batch processing
- **REST API**: Flask API server for integration
- **Multiple Models**: Logistic Regression, Linear SVM, Random Forest (all NLP-based)
- **Semantic Understanding**: Uses sentence transformers for better text comprehension
- **Topic Modeling**: LDA for additional semantic features

## Technology Stack

### Backend
- **Python 3.10+**
- **scikit-learn**: Machine learning models
- **sentence-transformers**: Semantic embeddings (all-MiniLM-L6-v2)
- **Flask**: REST API server
- **NLTK**: Text preprocessing
- **NumPy & Pandas**: Data processing

### Frontend
- **React**: Modern web interface
- **JavaScript/ES6**: Frontend logic

### Machine Learning
- **Feature Engineering**: Sentence transformers (384 dims) + LDA topic modeling (20 dims) = 404 total features
- **Models**: Logistic Regression, Linear SVM, Random Forest
- **Dataset**: 20 Newsgroups dataset (18,269 samples, 20 categories)

## Project Structure

```
NewsTopicClassifier/
├── data/                      # Dataset storage and features
├── models/                    # Trained models and embeddings
├── src/
│   ├── features/              # Feature engineering
│   │   ├── build_features_nlp.py    # NLP-based feature extraction
│   │   └── build_features.py        # TF-IDF feature extraction (legacy)
│   ├── models/               # Model training
│   │   ├── train_nlp.py      # Train NLP-based models
│   │   └── train.py          # Train TF-IDF models (legacy)
│   ├── api/                  # API server
│   │   └── app.py            # Flask REST API
│   └── cli/                  # CLI tools
│       └── predict_nlp.py    # Command-line prediction
├── frontend/                 # React frontend application
├── notebooks/                # Jupyter notebooks for exploration
├── src/experiments/          # Training results and visualizations
├── requirements.txt          # Python dependencies
├── setup.sh                  # Automated setup script
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 14+ and npm (for frontend)
- pip (Python package manager)

### Quick Setup

Run the automated setup script:

```bash
bash setup.sh
```

This will:
- Install all Python dependencies
- Download NLTK data
- Install frontend dependencies

### Manual Setup

#### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Frontend Setup

```bash
cd frontend
npm install
```

## Usage

### 1. Build Features

Generate semantic embeddings and topic features:

```bash
python src/features/build_features_nlp.py
```

This will:
- Download the 20 Newsgroups dataset
- Generate sentence transformer embeddings (384 dimensions)
- Create LDA topic features (20 dimensions)
- Save features to `data/` directory

### 2. Train Models

Train NLP-based classification models:

```bash
python src/models/train_nlp.py
```

This trains:
- Logistic Regression (NLP)
- Linear SVM (NLP)
- Random Forest (NLP)

Expected accuracy: ~74% on validation set

### 3. Web Interface

Start the API server:

```bash
python src/api/app.py
```

In another terminal, start the frontend:

```bash
cd frontend
npm start
```

Open your browser to `http://localhost:3000`

### 4. CLI Usage

```bash
# Predict with default model (Logistic Regression NLP)
python src/cli/predict_nlp.py "Your news article text here"

# Use specific model
python src/cli/predict_nlp.py "Your news article text here" --model linear_svm_nlp

# Show top 5 predictions
python src/cli/predict_nlp.py "Your news article text here" --top-k 5
```

### 5. API Usage

```bash
# Health check
curl http://localhost:5001/api/health

# Get available models
curl http://localhost:5001/api/models

# Predict topic
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Lakers won the championship game", "model": "linear_svm_nlp"}'
```

## Models

### Available Models

- **Logistic Regression (NLP)**: Fast, interpretable, ~74% accuracy
- **Linear SVM (NLP)**: Best performance, ~74.27% accuracy
- **Random Forest (NLP)**: Ensemble method, ~70% accuracy

All models use semantic embeddings (404 features) instead of traditional TF-IDF (30,000 features).

## Performance

- **Feature Dimensions**: 404 (vs 30,000 TF-IDF)
- **Model Size**: ~200KB per model (vs 14MB for TF-IDF models)
- **Training Time**: ~30 seconds (vs several minutes)
- **Accuracy**: ~74% on 20-class classification
- **Inference Speed**: <100ms per prediction

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `GET /api/categories` - List all categories
- `POST /api/predict` - Predict topic for text

## Dataset

Uses the **20 Newsgroups dataset** with 20 categories:
- Computer topics (graphics, hardware, OS, etc.)
- Recreation topics (autos, motorcycles, sports)
- Science topics (cryptography, electronics, medicine, space)
- Politics and religion topics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
