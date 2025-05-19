import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_predictions(model, test_loader, device):
    """Basic evaluation function that can be used by any model"""
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
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions = (predictions > 0.5).astype(int)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
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