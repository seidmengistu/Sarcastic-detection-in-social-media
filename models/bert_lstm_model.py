import os
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
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=Config.BERT_MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(label)
        }


class BertLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        
        # Freeze BERT layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):
            for param in self.bert.encoder.layer[i].parameters():
              param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc1    = nn.Linear(Config.LSTM_HIDDEN_SIZE * 2, Config.INTERMEDIATE_SIZE)
        self.act    = nn.ReLU()
        self.drop   = nn.Dropout(Config.DROPOUT_RATE)
        self.fc_out = nn.Linear(Config.INTERMEDIATE_SIZE, 1)
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_emb = bert_out.last_hidden_state     

        
        _, (hidden, _) = self.lstm(seq_emb)
        h_forward  = hidden[-2]   
        h_backward = hidden[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)

        
        x = self.fc1(h_cat)
        x = self.act(x)
        x = self.drop(x)
        logit = self.fc_out(x)
        return logit.view(-1)     


def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    #lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    for epoch in range(1, Config.NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_labels = [], []
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            ids   = batch['input_ids'].to(Config.DEVICE)
            mask  = batch['attention_mask'].to(Config.DEVICE)
            label = batch['label'].to(Config.DEVICE)

            logits = model(ids, mask)
            loss   = criterion(logits, label)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Collect training predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().tolist()
            all_train_preds.extend(preds)
            all_train_labels.extend(label.cpu().tolist())

        avg_train = total_train_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"  → Avg training loss: {avg_train:.4f}")
        
        # Store training metrics
        tr_report = classification_report(
            all_train_labels, 
            all_train_preds,
            target_names=['Not Sarcastic','Sarcastic'],
            output_dict=True
        )
        tr_report.update({
            'true': all_train_labels,
            'pred': all_train_preds
        })
        train_metrics.append(tr_report)

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds, all_lbls = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch['input_ids'].to(Config.DEVICE)
                mask  = batch['attention_mask'].to(Config.DEVICE)
                label = batch['label'].to(Config.DEVICE)

                logits = model(ids, mask)
                loss   = criterion(logits, label)
                total_val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().tolist()
                all_preds.extend(preds)
                all_lbls.extend(label.cpu().tolist())

        avg_val = total_val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"  → Avg validation loss: {avg_val:.4f}")

        # Store validation metrics
        v_report = classification_report(
            all_lbls, 
            all_preds,
            target_names=['Not Sarcastic','Sarcastic'],
            output_dict=True
        )
        v_report.update({
            'true': all_lbls,
            'pred': all_preds
        })
        val_metrics.append(v_report)


        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), Config.BERT_BEST_MODEL_PATH)
            print("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered - validation loss not improving")
                break

        print(classification_report(
            all_lbls, all_preds,
            target_names=['Not Sarcastic','Sarcastic'],
            digits=4
        ))

    # Plot training results
    from utils.analysis_data import plot_training_results
    plot_training_results(
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        save_path='bert_bilstm_training.png'
    )
    print("✓ Training plot saved as bert_bilstm_training.png")

    return model

def run():
    try:
        data = load_data(preprocessed=True)
        if not data:
            return False
        
        # convert string labels to ints
        for split in ['train','val','test']:
            texts, lbls = data[split]
            if isinstance(lbls[0], str):
                data[split] = (texts, [1 if l=='sarc' else 0 for l in lbls])
        
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        model = BertLSTMModel().to(Config.DEVICE)
        
        loaders = {}
        for name in ['train','val','test']:
            ds = SarcasmDataset(*data[name], tokenizer)
            shuffle = True if name=='train' else False
            loaders[name] = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle)
        
        train(model, loaders['train'], loaders['val'])
        
        model.load_state_dict(torch.load(Config.BERT_BEST_MODEL_PATH))
        evaluate_model_predictions(model, loaders['test'], Config.DEVICE)
        return True
    except Exception as e:
        print(f"Error in BERT-LSTM model: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
        run()
