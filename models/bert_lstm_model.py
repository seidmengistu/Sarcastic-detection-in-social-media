import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils.config import Config
from utils.dataset_loader import load_data
from utils.analysis_data import plot_training_results

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class BertLSTMModel(nn.Module):
    def __init__(self):
        super(BertLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        
        # Freeze BERT layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(10):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Model layers
        self.bert_norm = nn.LayerNorm(768)
        self.lstm = nn.LSTM(768, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATE)
        self.dropout2 = nn.Dropout(Config.DROPOUT_RATE + 0.1)
        self.layer_norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.bert_norm(bert_output.last_hidden_state)
        sequence_output = self.dropout1(sequence_output)
        
        # LSTM
        lstm_output, (hidden, _) = self.lstm(sequence_output)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classification
        hidden = self.layer_norm(hidden)
        hidden = self.dropout2(hidden)
        return self.sigmoid(self.fc(hidden))

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1), labels) / Config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
    
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
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels)
            
            total_loss += loss.item()
            preds = (outputs.view(-1) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    report_dict = classification_report(
        all_labels, 
        all_preds,
                                       target_names=['Not Sarcastic', 'Sarcastic'],
        output_dict=True
    )
    report_dict.update({'true': all_labels, 'pred': all_preds})
    
    return total_loss / len(dataloader), report_dict

def print_metrics(metrics_dict):
    """Print metrics in a formatted table."""
    print("\nValidation Metrics:")
    print(" " * 16 + "precision   recall  f1-score   support")
    print("-" * 58)
    
    # Print per-class metrics
    for class_name in ["Not Sarcastic", "Sarcastic"]:
        metrics = metrics_dict[class_name]
        print(f"{class_name:16} {metrics['precision']:.2f}      {metrics['recall']:.2f}     {metrics['f1-score']:.2f}      {metrics['support']:4d}")
    
    print("\n" + "-" * 58)
    
    # Print aggregate metrics - Fixed to use correct keys and calculate total support
    total_support = sum(metrics_dict[c]['support'] for c in ["Not Sarcastic", "Sarcastic"])
    print(f"{'accuracy':16}" + " " * 21 + f"{metrics_dict['accuracy']:.2f}      {total_support:4d}")
    print(f"{'macro avg':16} {metrics_dict['macro avg']['precision']:.2f}      "
          f"{metrics_dict['macro avg']['recall']:.2f}     {metrics_dict['macro avg']['f1-score']:.2f}      {total_support:4d}")
    print(f"{'weighted avg':16} {metrics_dict['weighted avg']['precision']:.2f}      "
          f"{metrics_dict['weighted avg']['recall']:.2f}     {metrics_dict['weighted avg']['f1-score']:.2f}      {total_support:4d}")

def run():
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f"Found {n_gpu} GPU(s) available")
            for i in range(n_gpu):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory Usage:")
                print(f"Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f}MB")
                print(f"Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f}MB")
        print(f"Using device: {device}")

        # Load and prepare data
        data = load_data(preprocessed=True)
        print("we are here",data)
        if not data:
            return False

        # Convert labels
        label_map = {'notsarc': 0, 'sarc': 1}
        for split in ['train', 'val', 'test']:
            texts, labels = data[split]
            data[split] = (texts, [label_map.get(l, l) for l in labels])

        # Initialize model
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Prepare datasets and dataloaders
        datasets = {
            split: SarcasmDataset(texts, labels, tokenizer) 
            for split, (texts, labels) in data.items()
        }
        
        torch.cuda.empty_cache()
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=(split == 'train'),
                pin_memory=True,
                num_workers=Config.NUM_WORKERS
            )
            for split, dataset in datasets.items()
        }
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        
        # Training tracking
        best_val_loss = float('inf')
        early_stopping_patience = 2
        no_improve_epochs = 0
        metrics = {
            'train_losses': [], 'val_losses': [],
            'train_metrics': [], 'val_metrics': []
        }
        
        # Training loop
        print("\n=== Starting Training ===")
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
            
            train_loss = train_epoch(model, dataloaders['train'], optimizer, criterion, device)
            val_loss, val_report = evaluate(model, dataloaders['val'], criterion, device)
            
            # Store metrics
            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            _, train_report = evaluate(model, dataloaders['train'], criterion, device)
            metrics['train_metrics'].append(train_report)
            metrics['val_metrics'].append(val_report)
            
            # Print metrics
            print(f"\nTraining Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print_metrics(val_report)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print("✓ Saved best model checkpoint")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Plot results
        plot_training_results(**metrics, save_path='training_results.png')
        
        # Final evaluation
        print("\n=== Evaluating on Test Set ===")
        model.load_state_dict(torch.load('best_model.pt'))
        test_loss, test_report = evaluate(model, dataloaders['test'], criterion, device)
        print("\n=== Final Test Results ===")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print("\nDetailed Test Metrics:")
        print("="*50)
        print_metrics(test_report)
        print("="*50)

    except Exception as e:
        print(f"Error in BERT-LSTM model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
        run()