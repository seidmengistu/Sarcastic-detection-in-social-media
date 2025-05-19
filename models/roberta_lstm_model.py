import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.config import Config
from utils.dataset_loader import load_data
from sklearn.metrics import classification_report
from utils.evaluation_utils import evaluate_model_predictions

class SarcasmDatasetRoBERTa(Dataset):
    """Dataset class for sarcasm detection using RoBERTa"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.ROBERTA_MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class RoBERTaLSTMModel(nn.Module):
    """RoBERTa + LSTM model for sarcasm detection"""
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        # Load pre-trained RoBERTa with specific configuration
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False
        )
        
        # Add LSTM layer
        self.lstm = nn.LSTM(
            input_size=Config.ROBERTA_HIDDEN_SIZE,
            hidden_size=Config.ROBERTA_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        
        # Add intermediate layer
        self.intermediate = nn.Linear(
            Config.ROBERTA_HIDDEN_SIZE * 2,
            Config.ROBERTA_INTERMEDIATE_SIZE
        )
        
        # Add dropout
        self.dropout = nn.Dropout(Config.ROBERTA_DROPOUT)
        
        # Add classifier
        self.classifier = nn.Linear(Config.ROBERTA_INTERMEDIATE_SIZE, 1)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Process through LSTM
        lstm_output, _ = self.lstm(outputs.last_hidden_state)
        
        # Get sequence output
        sequence_output = lstm_output[:, -1, :]
        
        # Process through intermediate layer
        intermediate_output = self.activation(self.intermediate(sequence_output))
        
        # Apply dropout
        intermediate_output = self.dropout(intermediate_output)
        
        # Get final output (remove sigmoid activation)
        logits = self.classifier(intermediate_output)
        
        return logits  # Return logits instead of sigmoid

def train_model(model, train_loader, val_loader, device):
    """Training function for RoBERTa-LSTM model"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.ROBERTA_LEARNING_RATE,
        weight_decay=0.01  
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Training]')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1), labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            del input_ids, attention_mask, labels, outputs
            torch.cuda.empty_cache()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Validation]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
                predictions = (predictions > 0.5).astype(int)
                
                loss = criterion(outputs.view(-1), batch['label'].to(device))
                val_loss += loss.item()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch + 1}:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        print("\nValidation Metrics:")
        print(classification_report(
            all_labels, 
            all_predictions,
            target_names=['Not Sarcastic', 'Sarcastic'],
            digits=2
        ))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.ROBERTA_BEST_MODEL_PATH)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

def run():
    """Main function to run the RoBERTa-LSTM model"""
    try:
        # Load and preprocess data
        data = load_data(preprocessed=True)
        if not data:
            raise ValueError("Could not load data")
        
        # Prepare tokenizer and datasets
        tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_MODEL_NAME)
        
        # Get data splits
        train_texts, train_labels = data['train']
        val_texts, val_labels = data['val']
        test_texts, test_labels = data['test']  # Add test data
        
        # Convert labels
        train_labels = [1 if label == 'sarc' else 0 for label in train_labels]
        val_labels = [1 if label == 'sarc' else 0 for label in val_labels]
        test_labels = [1 if label == 'sarc' else 0 for label in test_labels]  # Convert test labels
        
        # Create datasets
        train_dataset = SarcasmDatasetRoBERTa(train_texts, train_labels, tokenizer)
        val_dataset = SarcasmDatasetRoBERTa(val_texts, val_labels, tokenizer)
        test_dataset = SarcasmDatasetRoBERTa(test_texts, test_labels, tokenizer)  # Create test dataset
        
        # Create dataloaders with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(  # Create test loader
            test_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=True
        )
        
        # Initialize model
        model = RoBERTaLSTMModel()
        model = model.to(Config.DEVICE)
        
        # Train model
        train_model(model, train_loader, val_loader, Config.DEVICE)
        
        # Load best model for testing
        model.load_state_dict(torch.load(Config.ROBERTA_BEST_MODEL_PATH))
        
        # Evaluate
        results = evaluate_model_predictions(model, test_loader, Config.DEVICE)
        
        return True
        
    except Exception as e:
        print(f"Error in RoBERTa-LSTM model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run() 