import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils.config import Config
from utils.dataset_loader import load_data

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
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return dictionary of inputs
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BertLSTMModel(nn.Module):
    """BERT + LSTM model for sarcasm detection"""
    def __init__(self):
        super().__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Add LSTM layer with Config sizes
        self.lstm = nn.LSTM(
            input_size=768,  # BERT output size
            hidden_size=Config.HIDDEN_SIZE,  # From config
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
        
        lstm_output, (hidden, _) = self.lstm(bert_output.last_hidden_state)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        intermediate = self.activation(self.intermediate(hidden))
        
        output = self.dropout(intermediate)
        return torch.sigmoid(self.classifier(output))

def train_model(model, train_loader, val_loader, device):
    """Training function"""
    # Initialize training
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
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
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
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
        #  Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        #  Load and prepare data
        data = load_data(preprocessed=True)
        if not data:
            return False
        
        #  Convert labels to integers
        for split in ['train', 'val', 'test']:
            texts, labels = data[split]
            if isinstance(labels[0], str):
                labels = [1 if label == 'sarc' else 0 for label in labels]
                data[split] = (texts, labels)
        
        #  Initialize model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(device)
        
        #  Create datasets
        train_dataset = SarcasmDataset(data['train'][0], data['train'][1], tokenizer)
        val_dataset = SarcasmDataset(data['val'][0], data['val'][1], tokenizer)
        
        #  Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE
        )
        
        # Train model
        train_model(model, train_loader, val_loader, device)
        return True
        
    except Exception as e:
        print(f"Error in BERT-LSTM model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
        run()