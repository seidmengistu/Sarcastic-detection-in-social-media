# Only keep necessary imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def extract_metrics_from_report(report_str):
    """Extract f1-score from classification report string"""
    if isinstance(report_str, str):
        # Find the line with macro avg
        for line in report_str.split('\n'):
            if 'macro avg' in line:
                # Split the line and get f1-score (3rd value)
                values = [v for v in line.split() if v]
                return float(values[3])
    return report_str

def plot_training_results(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """Plot training results and metrics"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot F1 scores - Handle both dict and string metrics
        train_f1 = [extract_metrics_from_report(m) for m in train_metrics]
        val_f1 = [extract_metrics_from_report(m) for m in val_metrics]
        
        ax2.plot(epochs, train_f1, 'b-', label='Training F1')
        ax2.plot(epochs, val_f1, 'r-', label='Validation F1')
        ax2.set_title('F1 Score Progress')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        
        # For testing, we'll skip confusion matrix and class-wise plots if data isn't in right format
        if isinstance(val_metrics[-1], dict) and 'true' in val_metrics[-1]:
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
            
            if all(c in val_metrics[-1] for c in classes):
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
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        import traceback
        traceback.print_exc()

def get_metrics_report(y_true, y_pred, target_names=["Not Sarcastic", "Sarcastic"]):
    """Generate detailed metrics report"""
    return classification_report(y_true, y_pred, target_names=target_names)

def test_plotting():
    """Test function to verify plotting works"""
    print("Testing plotting functionality...")
    
    # Generate sample data
    n_epochs = 5
    train_losses = np.random.rand(n_epochs) * 0.5
    val_losses = np.random.rand(n_epochs) * 0.5
    
    # Create sample metrics
    train_metrics = [
        f"precision    recall  f1-score   support\nmacro avg  0.{i}6   0.{i}6    0.{i}6   1000\n"
        for i in range(n_epochs)
    ]
    val_metrics = [
        f"precision    recall  f1-score   support\nmacro avg  0.{i}7   0.{i}7    0.{i}7   1000\n"
        for i in range(n_epochs)
    ]
    
    try:
        plot_training_results(train_losses, val_losses, train_metrics, val_metrics)
        print("✓ Plotting test successful")
        return True
    except Exception as e:
        print(f"✗ Plotting test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_plotting()