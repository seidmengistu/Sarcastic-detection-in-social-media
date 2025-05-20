import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utils.config import Config
from models.bert_lstm_model import BertLSTMModel, SarcasmDataset
from models.roberta_lstm_model import RoBERTaLSTMModel, SarcasmDatasetRoBERTa
from transformers import BertTokenizer, RobertaTokenizer

def evaluate_test_set(model, test_loader, device):
    """
    Evaluate model on the test set
    """
    try:
        model.eval()
        all_predictions = []
        all_labels = []
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs).squeeze()
                if predictions.dim() == 0:  # Handle single prediction
                    predictions = predictions.unsqueeze(0)
                predictions = (predictions.cpu().numpy() > 0.5).astype(int)
                
                all_predictions.extend(predictions.tolist())  # Convert to list before extending
                all_labels.extend(labels.tolist())  # Convert to list before extending
        
        # Calculate and display metrics
        print("\nValidation Metrics:")
        print(classification_report(
            all_labels, 
            all_predictions,
            target_names=['Not Sarcastic', 'Sarcastic'],
            digits=2
        ))
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Sarcastic', 'Sarcastic'],
                   yticklabels=['Not Sarcastic', 'Sarcastic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('test_confusion_matrix.png')
        plt.close()
        
        return classification_report(
            all_labels, 
            all_predictions,
            target_names=['Not Sarcastic', 'Sarcastic'],
            digits=2,
            output_dict=True
        )
        
    except Exception as e:
        print(f"Error in test evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model_type="bert"):
    """Evaluate either BERT or RoBERTa model"""
    if model_type == "bert":
        model = BertLSTMModel()
        tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        dataset_class = SarcasmDataset
        model_path = Config.BERT_BEST_MODEL_PATH
    else:  # roberta
        model = RoBERTaLSTMModel()
        tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_MODEL_NAME)
        dataset_class = SarcasmDatasetRoBERTa
        model_path = Config.ROBERTA_BEST_MODEL_PATH
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(Config.DEVICE)
    
    from utils.dataset_loader import load_data
    data = load_data(preprocessed=True)
    test_texts, test_labels = data['test']
    test_labels = [1 if label == 'sarc' else 0 for label in test_labels]
    
    # Create test dataset and loader
    test_dataset = dataset_class(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate_test_set(model, test_loader, Config.DEVICE)
    
    # Print results
    print(f"\n=== {model_type.upper()} Model Test Results ===")
    print("=" * 50)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nTest Set Results:")
    print("-" * 50)
    print(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bert', 'roberta'], default='bert')
    args = parser.parse_args()
    
    evaluate_model(args.model)