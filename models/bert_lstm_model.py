import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils.config import Config
from utils.dataset_loader import load_data
from utils.evaluation_utils import evaluate_model_predictions


class SarcasmDataset(Dataset):
    """Dataset class for sarcasm detection"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert text to BERT input format
        text = str(self.texts[idx])
        label = int(self.labels[idx]) if isinstance(self.labels[idx], str) else self.labels[idx]
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=Config.BERT_MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Return dictionary of inputs
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BertLSTMModel(nn.Module):
    """BERT + LSTM model for sarcasm detection"""
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)

        # Add LSTM layer with Config sizes
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=Config.HIDDEN_SIZE,  
            batch_first=True,
            bidirectional=True
        )
        
        # Add intermediate layer
        self.intermediate = nn.Linear(Config.HIDDEN_SIZE * 2, Config.INTERMEDIATE_SIZE)
        self.activation = nn.ReLU()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
        # Final classification layer
        self.classifier = nn.Linear(Config.INTERMEDIATE_SIZE, 1)
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        _, (hidden, _) = self.lstm(bert_output.last_hidden_state)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        intermediate = self.activation(self.intermediate(hidden))
        
        output = self.dropout(intermediate)
        return self.classifier(output)

def train_model(model, train_loader, val_loader):
    """Training function"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # ====== Training Phase ======
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Clear gradients
            optimizer.zero_grad()
            
            # Move batch to device (GPU/CPU)
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # ====== Validation Phase ======
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['label'].to(Config.DEVICE)
                
                # Get predictions
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1), labels)
                
                # Store results
                val_loss += loss.item()
                predictions.extend((outputs.view(-1) > 0.5).int().cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.BERT_BEST_MODEL_PATH)
            print("✓ Saved best model")
        
        # Print metrics
        report = classification_report(
            true_labels, 
            predictions,
            target_names=['Not Sarcastic', 'Sarcastic']
        )
        print("\nValidation Metrics:")
        print(report)

def run():
    """Main function to run the model"""
    try:
        data = load_data(preprocessed=True)
        if not data:
            return False
        
        for split in ['train', 'val', 'test']:
            texts, labels = data[split]
            if isinstance(labels[0], str):
                labels = [1 if label == 'sarc' else 0 for label in labels]
                data[split] = (texts, labels)
        
        #  Initialize model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(Config.DEVICE)
        
        #  Create datasets
        train_dataset = SarcasmDataset(data['train'][0], data['train'][1], tokenizer)
        val_dataset = SarcasmDataset(data['val'][0], data['val'][1], tokenizer)
        test_dataset = SarcasmDataset(data['test'][0], data['test'][1], tokenizer)

        #  Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            # num_workers=Config.NUM_WORKERS,#uncomment those lines if you want to use GPU
            # pin_memory=Config.PIN_MEMORY   
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE,
            # num_workers=Config.NUM_WORKERS,#uncomment those lines if you want to use GPU
            # pin_memory=Config.PIN_MEMORY
        )
        
        # Create test dataset and loader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE,
            # num_workers=Config.NUM_WORKERS,#uncomment those lines if you want to use GPU
            # pin_memory=Config.PIN_MEMORY
        )
        
        # Train model
        train_model(model, train_loader, val_loader)
        
        # Load best model for testing
        model.load_state_dict(torch.load(Config.BERT_BEST_MODEL_PATH))
        
        # Evaluate on test set
        results = evaluate_model_predictions(model, test_loader, Config.DEVICE)
        
        return True
        
    except Exception as e:
        print(f"Error in BERT-LSTM model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
        run()