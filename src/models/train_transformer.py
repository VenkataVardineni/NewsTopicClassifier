"""
Training script for DistilBERT transformer model.
This provides state-of-the-art performance for text classification.
"""

import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset
import torch

class NewsDataset(Dataset):
    """Dataset class for news articles."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_text_data():
    """Load raw text data for transformer training."""
    print("Loading text data...")
    
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from src.features.build_features import clean_text
    
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
    
    # Combine and clean
    all_texts = newsgroups_train.data + newsgroups_test.data
    all_labels = list(newsgroups_train.target) + list(newsgroups_test.target)
    
    # Clean texts (minimal cleaning for transformers)
    cleaned_texts = []
    cleaned_labels = []
    for text, label in zip(all_texts, all_labels):
        cleaned = clean_text(text)
        if len(cleaned) > 10:  # Filter very short texts
            cleaned_texts.append(cleaned)
            cleaned_labels.append(label)
    
    # Split
    texts_train, texts_val, y_train, y_val = train_test_split(
        cleaned_texts, cleaned_labels, test_size=0.2, random_state=42, stratify=cleaned_labels
    )
    
    category_names = newsgroups_train.target_names
    
    return texts_train, texts_val, y_train, y_val, category_names

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

def main():
    """Main training function for DistilBERT."""
    print("=" * 60)
    print("Training DistilBERT Transformer Model")
    print("=" * 60)
    print("\nNote: This will take 15-45 minutes depending on your hardware.")
    print("GPU is recommended but not required.\n")
    
    # Load data
    texts_train, texts_val, y_train_text, y_val_text, category_names = load_text_data()
    
    print(f"Training samples: {len(texts_train)}")
    print(f"Validation samples: {len(texts_val)}")
    print(f"Categories: {len(category_names)}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(category_names)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = NewsDataset(texts_train, y_train_text, tokenizer)
    val_dataset = NewsDataset(texts_val, y_val_text, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/distilbert_checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"\nValidation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1-Score: {eval_results['eval_f1']:.4f}")
    
    # Make predictions
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val_text, y_pred, target_names=category_names))
    
    # Save confusion matrix
    os.makedirs('src/experiments', exist_ok=True)
    cm_path = 'src/experiments/confusion_matrix_distilbert.png'
    plot_confusion_matrix(y_val_text, y_pred, category_names, 'DistilBERT', cm_path)
    
    # Save model and tokenizer
    print("\nSaving model...")
    model_save_path = 'models/distilbert'
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save category names
    with open('models/category_names.pkl', 'wb') as f:
        pickle.dump(category_names, f)
    
    print(f"\nModel saved to {model_save_path}")
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
