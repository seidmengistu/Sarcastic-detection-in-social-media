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

def objective(trial):
    # Load and prepare data first
    data = load_data(preprocessed=True)
    if not data:
        raise ValueError("Could not load data")
    
    # Convert string labels to ints if needed
    for split in ['train', 'val']:
        texts, lbls = data[split]
        if isinstance(lbls[0], str):
            data[split] = (texts, [1 if l=='sarc' else 0 for l in lbls])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    # Create data loaders
    train_dataset = SarcasmDataset(*data['train'], tokenizer)
    val_dataset = SarcasmDataset(*data['val'], tokenizer)
    
    # Define hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [256, 384, 512]),
        'intermediate_size': trial.suggest_categorical('intermediate_size', [128, 256]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.05),
        'frozen_layers': trial.suggest_int('frozen_layers', 6, 9),  # Number of BERT layers to freeze
    }
    
    # Create data loaders with trial batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False
    )
    
    # Create model with trial parameters
    model = BertLSTMModel()
    
    # Freeze BERT layers according to trial
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(params['frozen_layers']):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
            
    model = model.to(Config.DEVICE)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    print(f"\nTrial {trial.number + 1} Training Progress:")
    
    # Training loop
    for epoch in tqdm(range(Config.NUM_EPOCHS), desc="Epochs"):
        model.train()
        train_loss = 0
        
        # Add progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for batch in train_pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(Config.DEVICE)
            mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['label'].to(Config.DEVICE)
                
                outputs = model(ids, mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
        
        # Report to Optuna
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def find_best_hyperparameters(n_trials=6):
    print(f"\nRunning {n_trials} trials...")
    print("=" * 50)
    
    def print_trial_callback(study, trial):
        print(f"\nTrial {trial.number + 1}/{n_trials} completed")
        print(f"Current trial value (validation loss): {trial.value:.4f}")
        
        if study.best_trial == trial:
            print("✨ New best trial!")
            print(f"Best parameters so far:")
            for key, value in trial.params.items():
                print(f"  {key}: {value}")
        print("-" * 50)
    
    study = optuna.create_study(direction="minimize")
    
    # Wrap the optimization in a progress bar
    with tqdm(total=n_trials, desc="Total Trials") as pbar:
        def update_pbar_callback(study, trial):
            pbar.update(1)
            print_trial_callback(study, trial)
            
        study.optimize(objective, n_trials=n_trials, callbacks=[update_pbar_callback])
    
    print("\nOptimization Complete!")
    print("=" * 50)
    print("\nBest hyperparameters found:")
    print("-" * 40)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.4f}")
    
    # Save best parameters
    with open("best_hyperparameters.txt", "w") as f:
        f.write("Best Hyperparameters:\n")
        f.write("-" * 40 + "\n")
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