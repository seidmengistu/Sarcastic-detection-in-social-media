import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertLSTMModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_hidden_size=256, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_hidden_size * 2, 1)  # *2 for bidirectional
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        pooled_output = torch.mean(lstm_output, dim=1)  # Average pooling
        pooled_output = self.dropout(pooled_output)
        return torch.sigmoid(self.classifier(pooled_output))

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            outputs = outputs.view(-1)
            
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            
            preds = (outputs > 0.5).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return (total_loss / len(dataloader), 
            classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))

def visualize_results(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """
    Visualize training results with multiple plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot F1 scores
    ax2.plot(epochs, [m['macro avg']['f1-score'] for m in train_metrics], 'b-', label='Training F1')
    ax2.plot(epochs, [m['macro avg']['f1-score'] for m in val_metrics], 'r-', label='Validation F1')
    ax2.set_title('F1 Score Progress')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    
    # Plot confusion matrix for last validation
    cm = confusion_matrix(val_metrics[-1]['true'], val_metrics[-1]['pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax3)
    ax3.set_title('Confusion Matrix (Last Validation)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Plot class-wise F1 scores
    classes = ['Not Sarcastic', 'Sarcastic']
    x = range(len(classes))
    width = 0.35
    ax4.bar([i - width/2 for i in x], 
            [val_metrics[-1][c]['f1-score'] for c in classes],
            width, label='Validation')
    ax4.bar([i + width/2 for i in x],
            [train_metrics[-1][c]['f1-score'] for c in classes],
            width, label='Training')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Class-wise F1 Scores (Last Epoch)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load data
        PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../."))
        PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "preprocessed_dataset.csv")
        
        print("Loading data...")
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        
        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertLSTMModel().to(device)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df['text'].values, df['class'].values, test_size=0.4, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
        val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer)
        test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)
        test_dataloader = DataLoader(test_dataset, batch_size=16)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        num_epochs = 3
        best_val_loss = float('inf')
        
        # Add lists to store metrics
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        # Training loop
        print("\n=== Starting Training ===")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
            val_loss, val_report = evaluate(model, val_dataloader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Get training metrics
            _, train_report = evaluate(model, train_dataloader, criterion, device)
            train_metrics.append(train_report)
            val_metrics.append(val_report)
            
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print("\nValidation Metrics:")
            print(val_report)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print("✓ Saved best model checkpoint")
        
        # Visualize results
        visualize_results(
            train_losses, 
            val_losses, 
            train_metrics, 
            val_metrics,
            save_path='training_results.png'
        )
        
        print("\n=== Training Completed ===")
        print("\n=== Evaluating on Test Set ===")
        
        # Load best model and evaluate on test set
        print("Loading best model for test evaluation...")
        model.load_state_dict(torch.load('best_model.pt'))
        test_loss, test_report = evaluate(model, test_dataloader, criterion, device)
        
        print("\n=== Final Test Results ===")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print("\nDetailed Test Metrics:")
        print("="*50)
        print(test_report)
        print("="*50)

    except Exception as e:
        print(f"Error in BERT+LSTM model: {str(e)}")
        import traceback
        traceback.print_exc()
