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
from models.roberta_lstm_model import RoBERTaLSTMModel, SarcasmDatasetRoBERTa
from transformers import BertTokenizer, RobertaTokenizer
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

def objective(trial, data, model_type="bert"):
    """Optuna objective function to minimize validation loss"""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]), 
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),  
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'intermediate_size': trial.suggest_categorical('intermediate_size', [64, 128, 256])  
    }
    
    # Set up model-specific components
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model_class = BertLSTMModel
        dataset_class = SarcasmDataset
        max_length = Config.BERT_MAX_LENGTH
        model_name = Config.BERT_MODEL_NAME
    else:  
        tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_MODEL_NAME)
        model_class = RoBERTaLSTMModel
        dataset_class = SarcasmDatasetRoBERTa
        max_length = Config.ROBERTA_MAX_LENGTH
        model_name = Config.ROBERTA_MODEL_NAME
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    
    train_labels = [1 if label == 'sarc' else 0 for label in train_labels]
    val_labels = [1 if label == 'sarc' else 0 for label in val_labels]
    
    train_dataset = dataset_class(train_texts, train_labels, tokenizer)
    val_dataset = dataset_class(val_texts, val_labels, tokenizer)
    
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
    
    model = model_class(model_name=model_name)
    if model_type == "roberta":
        model.roberta.gradient_checkpointing_enable()  
    
    # Update model architecture with smaller sizes
    model.lstm = nn.LSTM(
        input_size=768,
        hidden_size=params['hidden_size'],
        batch_first=True,
        bidirectional=True
    )
    model.intermediate = nn.Linear(params['hidden_size'] * 2, params['intermediate_size'])
    model.dropout = nn.Dropout(params['dropout_rate'])
    model.classifier = nn.Linear(params['intermediate_size'], 1)
    
    model = model.to(Config.DEVICE)
    
    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(3):
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

def find_best_hyperparameters(model_type="bert", n_trials=2):
    """Run hyperparameter optimization"""
    data = load_data_once()
    
    study = optuna.create_study(direction="minimize")
    
    # Create progress bar for trials
    with tqdm(total=n_trials, desc=f"Optimizing {model_type.upper()} Model") as progress_bar:
        def callback(study, trial):
            progress_bar.update(1)
            progress_bar.set_postfix({
                'trial': f'{trial.number + 1}/{n_trials}',
                'best_loss': f'{study.best_value:.4f}'
            })
        
        study.optimize(
            lambda trial: objective(trial, data, model_type), 
            n_trials=n_trials, 
            callbacks=[callback]
        )
    
    # Save best parameters to text file
    filename = f"best_hyperparameters_{model_type}.txt"
    with open(filename, "w") as f:
        f.write(f"Best Hyperparameters for {model_type.upper()}:\n")
        f.write("-" * 20 + "\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest validation loss: {study.best_value:.4f}")
    
    return study.best_params

if __name__ == "__main__":
    print("\nStarting Hyperparameter Optimization")
    print("=" * 40)
    
    print("\nOptimizing BERT-LSTM model...")
    bert_params = find_best_hyperparameters("bert")
    
    print("\nOptimizing RoBERTa-LSTM model...")
    roberta_params = find_best_hyperparameters("roberta")
    
    print("\nOptimization Complete!")
    print("=" * 40)
    
    print("\nBest BERT hyperparameters:")
    for key, value in bert_params.items():
        print(f"{key}: {value}")
        
    print("\nBest RoBERTa hyperparameters:")
    for key, value in roberta_params.items():
        print(f"{key}: {value}") 