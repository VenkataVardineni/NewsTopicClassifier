# Quick Start Guide

## First Time Setup

1. **Install dependencies:**
   ```bash
   bash setup.sh
   ```

2. **Build features and train models:**
   ```bash
   # Download dataset and create features
   python src/features/build_features.py
   
   # Train models
   python src/models/train.py
   ```

## Running the Application

### Option 1: CLI Only

```bash
python src/cli/predict.py "The Lakers won the championship game last night with a score of 108-95."
```

### Option 2: Web Interface

**Terminal 1 - Start API:**
```bash
python src/api/app.py
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm start
```

Then open `http://localhost:3000` in your browser.

## Example Predictions

Try these sample texts:

**Sports:**
```
The Lakers defeated the Warriors 108-95 in a thrilling game last night. 
LeBron James scored 35 points and had 10 assists.
```

**Technology:**
```
Apple announced its new iPhone 15 with advanced AI features and improved camera system. 
The device will be available next month.
```

**Politics:**
```
The Senate passed a new bill today that will affect healthcare policies across the nation. 
The vote was 52-48 in favor.
```

## Troubleshooting

- **Model not found:** Make sure you've run `python src/models/train.py` first
- **API connection error:** Ensure the API server is running on port 5000
- **NLTK errors:** Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

