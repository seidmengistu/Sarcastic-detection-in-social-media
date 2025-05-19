import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from .config import Config

def plot_training_results(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """Plot training results."""
    try:
        if not all([train_losses, val_losses, train_metrics, val_metrics]):
            print("Error: Empty metrics provided")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=Config.FIGURE_SIZE, dpi=Config.DPI)
        
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
        
        # Plot confusion matrix
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
        plt.close()
        
    except Exception as e:
        print(f"Error plotting results: {str(e)}")

def get_metrics_report(y_true, y_pred):
    """Generate detailed metrics report."""
    try:
        return classification_report(
            y_true, 
            y_pred,
            target_names=['Not Sarcastic', 'Sarcastic'],
            output_dict=True
        )
    except Exception as e:
        print(f"Error generating metrics report: {str(e)}")
        return None

if __name__ == "__main__":
    print("This module provides plotting and metrics functions for model evaluation")