import optuna
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert_lstm_model import BertLSTMModel, SarcasmDataset
from transformers import BertTokenizer
from utils.config import Config
from utils.dataset_loader import load_data
import torch.nn as nn

def load_data_once():
    """Load data once and return it"""
    print("\nLoading dataset...")
    data = load_data(preprocessed=True)
    if not data:
        raise ValueError("Could not load data")
    return data

def objective(trial, data):
    """Optuna objective function to minimize validation loss"""
    # Reduce parameter ranges for memory constraints
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8]),  # Reduced batch sizes
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256]),  # Reduced hidden sizes
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'intermediate_size': trial.suggest_categorical('intermediate_size', [64, 128]),  # Reduced intermediate sizes
        'lstm_layers': 1,  # Fixed to 1 layer
        'lstm_dropout': 0.0  # No LSTM dropout needed for single layer
    }
    
    # Set up BERT components
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    model_name = Config.BERT_MODEL_NAME
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    
    train_labels = [1 if label == 'sarc' else 0 for label in train_labels]
    val_labels = [1 if label == 'sarc' else 0 for label in val_labels]
    
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
    val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'],
        pin_memory=True
    )
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = BertLSTMModel(model_name=model_name)
    
    # Update model architecture
    model.lstm = nn.LSTM(
        input_size=768,
        hidden_size=params['hidden_size'],
        num_layers=params['lstm_layers'],
        batch_first=True,
        bidirectional=True
    )
    
    model.intermediate = nn.Linear(params['hidden_size'] * 2, params['intermediate_size'])
    model.dropout = nn.Dropout(params['dropout_rate'])
    model.classifier = nn.Linear(params['intermediate_size'], 1)
    
    model = model.to(Config.DEVICE)
    
    # Use gradient checkpointing
    model.bert.gradient_checkpointing_enable()
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1), labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            with autocast(device_type='cuda'):
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(Config.DEVICE)
                    attention_mask = batch['attention_mask'].to(Config.DEVICE)
                    labels = batch['label'].to(Config.DEVICE)
                    
                    outputs = model(input_ids, attention_mask)
                    val_loss += criterion(outputs.view(-1), labels).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def find_best_hyperparameters(n_trials=5):
    """Run hyperparameter optimization"""
    data = load_data_once()
    
    study = optuna.create_study(direction="minimize")
    
    # Create progress bar for trials
    with tqdm(total=n_trials, desc="Optimizing BERT Model") as progress_bar:
        def callback(study, trial):
            progress_bar.update(1)
            progress_bar.set_postfix({
                'trial': f'{trial.number + 1}/{n_trials}',
                'best_loss': f'{study.best_value:.4f}'
            })
        
        study.optimize(
            lambda trial: objective(trial, data), 
            n_trials=n_trials, 
            callbacks=[callback]
        )
    
    # Save best parameters to text file
    filename = "best_hyperparameters_bert.txt"
    with open(filename, "w") as f:
        f.write("Best Hyperparameters for BERT:\n")
        f.write("-" * 20 + "\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest validation loss: {study.best_value:.4f}")
    
    return study.best_params

if __name__ == "__main__":
    print("\nStarting Hyperparameter Optimization")
    print("=" * 40)
    
    print("\nOptimizing BERT-LSTM model...")
    bert_params = find_best_hyperparameters()
    
    print("\nOptimization Complete!")
    print("=" * 40)
    
    print("\nBest BERT hyperparameters:")
    for key, value in bert_params.items():
        print(f"{key}: {value}") 