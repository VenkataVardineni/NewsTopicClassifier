# Setup Instructions

Complete setup guide for News Topic Classifier.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** (check with `python3 --version`)
- **Node.js 14+ and npm** (check with `node --version` and `npm --version`)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd NewsTopicClassifier
```

## Step 2: Automated Setup (Recommended)

Run the automated setup script:

```bash
bash setup.sh
```

This script will:
1. Check Python version
2. Install all Python dependencies
3. Download NLTK data
4. Install frontend dependencies

## Step 3: Manual Setup (Alternative)

If you prefer manual setup or the script fails:

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or use pip3 if pip points to Python 2
pip3 install -r requirements.txt

# Download required NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Frontend Setup

```bash
cd frontend
npm install

# If npm install fails, try:
npm install --legacy-peer-deps
```

## Step 4: Build Features

Generate the semantic embeddings and topic features:

```bash
python src/features/build_features_nlp.py
```

**Note**: This step will:
- Download the 20 Newsgroups dataset (~20MB)
- Generate sentence transformer embeddings (takes ~5 minutes)
- Create LDA topic features
- Save everything to `data/` and `models/` directories

**Expected output**: 
- Feature files in `data/` directory
- Sentence transformer model in `models/sentence_transformer/`
- LDA models in `models/lda_model.pkl` and `models/lda_vectorizer.pkl`

## Step 5: Train Models

Train the classification models:

```bash
python src/models/train_nlp.py
```

**Note**: This step will:
- Train 3 models (Logistic Regression, Linear SVM, Random Forest)
- Take ~30 seconds to 1 minute
- Save models to `models/` directory
- Generate confusion matrices in `src/experiments/`

**Expected output**:
- `models/logistic_regression_nlp.pkl`
- `models/linear_svm_nlp.pkl`
- `models/random_forest_nlp.pkl`
- Confusion matrix images in `src/experiments/`

## Step 6: Verify Installation

Test the CLI to verify everything works:

```bash
python src/cli/predict_nlp.py "The Lakers won the championship game last night"
```

You should see output like:
```
Predicted Topic: rec.sport.hockey
Category ID: 10

Top 3 Predictions:
  rec.sport.hockey                   0.9201
  rec.sport.baseball                 0.0463
  talk.politics.misc                 0.0064
```

## Step 7: Run the Application

### Start the API Server

```bash
python src/api/app.py
```

The API will start on `http://localhost:5001`

### Start the Frontend (in a new terminal)

```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000`

### Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## Troubleshooting

### Python Version Issues

If you get Python version errors:
```bash
# Check your Python version
python3 --version

# Use python3 explicitly
python3 src/features/build_features_nlp.py
```

### Permission Errors

If you get permission errors during installation:
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### Port Already in Use

If port 5001 or 3000 is already in use:

**For API (port 5001):**
```bash
# Find and kill the process
lsof -ti:5001 | xargs kill -9

# Or change the port in src/api/app.py
```

**For Frontend (port 3000):**
```bash
# Find and kill the process
lsof -ti:3000 | xargs kill -9

# Or React will prompt you to use a different port
```

### NLTK Data Download Issues

If NLTK data fails to download:
```bash
python3
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> exit()
```

### Frontend Build Issues

If `npm install` fails:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

### Model Not Found Errors

If you get "Model not found" errors:
1. Make sure you've run `python src/features/build_features_nlp.py`
2. Make sure you've run `python src/models/train_nlp.py`
3. Check that model files exist in `models/` directory

## File Structure After Setup

After complete setup, you should have:

```
NewsTopicClassifier/
├── data/
│   ├── X_train.pkl
│   ├── X_val.pkl
│   ├── y_train.pkl
│   ├── y_val.pkl
│   └── metadata.pkl
├── models/
│   ├── sentence_transformer/      # Sentence transformer model
│   ├── lda_model.pkl              # LDA topic model
│   ├── lda_vectorizer.pkl         # LDA vectorizer
│   ├── category_names.pkl         # Category labels
│   ├── logistic_regression_nlp.pkl
│   ├── linear_svm_nlp.pkl
│   └── random_forest_nlp.pkl
└── src/experiments/
    ├── confusion_matrix_*.png     # Confusion matrices
    └── training_results_nlp.pkl    # Training results
```

## Next Steps

1. **Try the web interface**: Open `http://localhost:3000` and test predictions
2. **Use the CLI**: Try different models and texts
3. **Explore the API**: Test the REST endpoints
4. **Check experiments**: View confusion matrices in `src/experiments/`

## System Requirements

- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for models and data
- **Internet**: Required for initial dataset download and model downloads

## Support

If you encounter issues not covered here:
1. Check the error messages carefully
2. Verify all prerequisites are installed
3. Ensure you've completed all setup steps in order
4. Check that model files exist in the `models/` directory

