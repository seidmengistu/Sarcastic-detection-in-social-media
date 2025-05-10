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
from utils.config import Config
from utils.dataset_loader import load_data
from utils.analysis_data import plot_training_results, get_metrics_report

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
    def __init__(self):
        super(BertLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)  # Use dropout rate from config
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Add layer normalization for better regularization
        self.layer_norm = nn.LayerNorm(512)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply dropout after BERT
        sequence_output = self.dropout(sequence_output)
        
        # Apply LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply layer normalization
        hidden = self.layer_norm(hidden)
        
        # Apply dropout before final classification
        hidden = self.dropout(hidden)
        
        # Apply linear layer and sigmoid
        output = self.fc(hidden)
        output = self.sigmoid(output)
        
        return output

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        # Ensure outputs and labels have compatible shapes
        outputs_flat = outputs.view(-1)
        labels_flat = labels.view(-1)
        
        loss = criterion(outputs_flat, labels_flat)
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
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Ensure outputs and labels have compatible shapes
            outputs_flat = outputs.view(-1)
            labels_flat = labels.view(-1)
            
            loss = criterion(outputs_flat, labels_flat)
            
            total_loss += loss.item()
            
            # Convert predictions to binary (0 or 1)
            preds = (outputs_flat > 0.5).int().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_flat.cpu().numpy().tolist())
    
    # Calculate metrics
    report_dict = classification_report(all_labels, all_preds, 
                                       target_names=['Not Sarcastic', 'Sarcastic'],
                                       output_dict=True)
    
    # Store true and predicted values for confusion matrix
    report_dict['true'] = all_labels
    report_dict['pred'] = all_preds
    
    return total_loss / len(dataloader), report_dict

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load data using utility function
        data = load_data(preprocessed=True)
        train_texts, train_labels = data['train']
        val_texts, val_labels = data['val']
        test_texts, test_labels = data['test']
        
        # Initialize model with config
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(device)
        
        # Create datasets
        train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
        val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer)
        test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
        
        # Create dataloaders using config
        train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
        
        # Training setup with weight decay (L2 regularization)
        criterion = nn.BCELoss()
        optimizer = AdamW(model.parameters(), 
                         lr=Config.LEARNING_RATE, 
                         weight_decay=Config.WEIGHT_DECAY)  # Use weight decay from config
        num_epochs = Config.NUM_EPOCHS
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
            
            # Print only the important metrics, not the full lists
            print_report = {k: v for k, v in val_report.items() if k not in ['true', 'pred']}
            print(f"Accuracy: {print_report['accuracy']:.4f}")
            print(f"F1 Score (macro): {print_report['macro avg']['f1-score']:.4f}")
            print(f"Precision: {print_report['macro avg']['precision']:.4f}")
            print(f"Recall: {print_report['macro avg']['recall']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print("✓ Saved best model checkpoint")
        
        # Use utility for visualization
        plot_training_results(
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

def test_model_pipeline():
    try:
        print("Testing BERT+LSTM pipeline...")
        
        # Test data loading
        data = load_data(preprocessed=True)
        train_texts, train_labels = data['train']
        print("✓ Data loading successful")
        print(f"Number of training examples: {len(train_texts)}")
        
        # Test model initialization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(device)
        print("✓ Model initialization successful")
        
        # Test forward pass with small batch
        dataset = SarcasmDataset(train_texts[:16], train_labels[:16], tokenizer)
        dataloader = DataLoader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        
        outputs = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device)
        )
        print("✓ Forward pass successful")
        print(f"Output shape: {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error in BERT+LSTM test: {str(e)}")
        traceback.print_exc()
        return False
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_pipeline()
    else:
        run()