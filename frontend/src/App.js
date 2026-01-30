import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [model, setModel] = useState('logistic_regression_nlp');
  const [availableModels, setAvailableModels] = useState(['logistic_regression', 'linear_svm']);

  // Fetch available models on component mount
  useEffect(() => {
    fetch(`${API_URL}/api/models`)
      .then(res => res.json())
      .then(data => {
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models);
        }
      })
      .catch(err => console.error('Failed to fetch models:', err));
  }, []);

  const getModelDisplayName = (modelName) => {
    const names = {
      'logistic_regression': 'Logistic Regression',
      'logistic_regression_nlp': 'Logistic Regression (NLP)',
      'linear_svm': 'Linear SVM',
      'linear_svm_nlp': 'Linear SVM (NLP)',
      'random_forest_nlp': 'Random Forest (NLP)',
      'xgboost': 'XGBoost',
      'xgboost_nlp': 'XGBoost (NLP)',
      'distilbert': 'DistilBERT (Transformer)'
    };
    return names[modelName] || modelName.replace('_nlp', ' (NLP)').replace(/_/g, ' ');
  };

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text to classify.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, model }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to predict');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>📰 News Topic Classifier</h1>
          <p className="subtitle">
            Classify news articles into categories using machine learning
          </p>
        </header>

        <div className="main-content">
          <div className="input-section">
            <div className="model-selector">
              <label htmlFor="model-select">Model:</label>
              <select
                id="model-select"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={loading}
              >
                {availableModels.map(modelName => (
                  <option key={modelName} value={modelName}>
                    {getModelDisplayName(modelName)}
                  </option>
                ))}
              </select>
            </div>

            <div className="text-input-wrapper">
              <label htmlFor="text-input">Enter news article text:</label>
              <textarea
                id="text-input"
                className="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your news article here..."
                rows="10"
                disabled={loading}
              />
              <div className="char-count">
                {text.length} characters
              </div>
            </div>

            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading || !text.trim()}
              >
                {loading ? 'Classifying...' : 'Classify Article'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </button>
            </div>
          </div>

          {error && (
            <div className="result-section error">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <div className="result-section">
              <h3>Prediction Result</h3>
              <div className="prediction-card">
                <div className="predicted-category">
                  <span className="category-label">Category:</span>
                  <span className="category-name">{result.category}</span>
                </div>
              </div>

              {result.top_predictions && result.top_predictions.length > 0 && (
                <div className="top-predictions">
                  <h4>Top Predictions</h4>
                  <div className="predictions-list">
                    {result.top_predictions.map((pred, index) => (
                      <div
                        key={index}
                        className={`prediction-item ${
                          index === 0 ? 'top-prediction' : ''
                        }`}
                      >
                        <span className="prediction-category">
                          {pred.category}
                        </span>
                        <div className="probability-bar-wrapper">
                          <div
                            className="probability-bar"
                            style={{
                              width: `${pred.probability * 100}%`,
                            }}
                          />
                        </div>
                        <span className="probability-value">
                          {(pred.probability * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <footer className="footer">
          <p>
            Powered by scikit-learn, XGBoost & Transformers
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
