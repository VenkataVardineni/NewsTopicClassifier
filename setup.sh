#!/bin/bash
# Setup script for News Topic Classifier

echo "Setting up News Topic Classifier..."
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Build features: python src/features/build_features.py"
echo "2. Train models: python src/models/train.py"
echo "3. Start API: python src/api/app.py"
echo "4. Start frontend: cd frontend && npm start"

