"""
Training script for news topic classification models.
Trains multiple models with hyperparameter tuning for optimal performance.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed features and labels."""
    print("Loading data...")
    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('models/category_names.pkl', 'rb') as f:
        category_names = pickle.load(f)
    
    return X_train, X_val, y_train, y_val, category_names

def plot_confusion_matrix(y_true, y_pred, category_names, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=category_names,
        yticklabels=category_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, category_names, use_tuning=False):
    """Train model and evaluate on validation set."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Hyperparameter tuning if requested
    if use_tuning and hasattr(model, 'get_params'):
        print("Performing hyperparameter tuning...")
        if 'Logistic Regression' in model_name:
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 2000]
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        elif 'SVM' in model_name:
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'loss': ['hinge', 'squared_hinge']
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
    
    # Train model
    print("Training...")
    model.fit(X_train, y_train)
    
    # Predictions
    print("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=category_names))
    
    # Save confusion matrix
    os.makedirs('src/experiments', exist_ok=True)
    cm_path = f'src/experiments/confusion_matrix_{model_name.lower().replace(" ", "_").replace("-", "_")}.png'
    plot_confusion_matrix(y_val, y_val_pred, category_names, model_name, cm_path)
    
    # Save model
    model_path = f'models/{model_name.lower().replace(" ", "_").replace("-", "_")}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    return {
        'model_name': model_name,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'model_path': model_path
    }

def main():
    """Main training function."""
    print("News Topic Classifier - Model Training (Improved)")
    print("=" * 60)
    
    # Load data
    X_train, X_val, y_train, y_val, category_names = load_data()
    
    # Convert sparse matrices to dense for XGBoost if needed
    X_train_dense = X_train.toarray() if issparse(X_train) else X_train
    X_val_dense = X_val.toarray() if issparse(X_val) else X_val
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('src/experiments', exist_ok=True)
    
    results = []
    
    # Train Improved Logistic Regression with optimized hyperparameters
    print("\nTraining improved Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
        C=10.0,
        solver='lbfgs',
        multi_class='multinomial'
    )
    lr_results = train_and_evaluate(
        lr_model, 'Logistic Regression',
        X_train, y_train, X_val, y_val, category_names, use_tuning=False
    )
    results.append(lr_results)
    
    # Train Improved Linear SVM with optimized hyperparameters
    print("\nTraining improved Linear SVM...")
    svm_base = LinearSVC(
        max_iter=2000,
        random_state=42,
        C=10.0,
        dual=False,
        loss='squared_hinge'
    )
    svm_model = CalibratedClassifierCV(svm_base, cv=3, method='sigmoid')
    svm_results = train_and_evaluate(
        svm_model, 'Linear SVM',
        X_train, y_train, X_val, y_val, category_names, use_tuning=False
    )
    results.append(svm_results)
    
    # Train XGBoost with optimized hyperparameters
    try:
        import xgboost as xgb
        print("\nTraining XGBoost (Best Model)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist'  # Faster training
        )
        xgb_results = train_and_evaluate(
            xgb_model, 'XGBoost',
            X_train_dense, y_train, X_val_dense, y_val, category_names, use_tuning=False
        )
        results.append(xgb_results)
    except ImportError:
        print("\nXGBoost not available, skipping...")
        print("Install with: pip install xgboost")
    
    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Val Accuracy':<15} {'Val F1-Score':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_name']:<25} {r['val_acc']:<15.4f} {r['val_f1']:<15.4f}")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['val_acc'])
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model['model_name']}")
        print(f"  Validation Accuracy: {best_model['val_acc']:.4f}")
        print(f"  Validation F1-Score: {best_model['val_f1']:.4f}")
        print(f"{'='*60}")
    
    # Save results
    results_path = 'src/experiments/training_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")
    
    print("\nTraining complete!")
    print("\nNote: For state-of-the-art results, train the transformer model:")
    print("  python src/models/train_transformer.py")

if __name__ == '__main__':
    main()
