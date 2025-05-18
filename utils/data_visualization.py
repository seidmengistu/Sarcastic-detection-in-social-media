import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from config import Config

def visualize_dataset_splits(save_path='dataset_distribution.pdf'):
    """
    Analyze and visualize the distribution of sarcastic and non-sarcastic samples
    in training, validation, and test sets.
    """
    try:
        print("Analyzing dataset distribution...")
        
        df = pd.read_csv(Config.PREPROCESSED_DATA_PATH)
        total_samples = len(df)
        
        train_val_df, test_df = train_test_split(
            df, 
            train_size=Config.TRAIN_VAL_SIZE,
            random_state=42,
            stratify=df['class']
        )
        
        train_df, val_df = train_test_split(
            train_val_df,
            train_size=1-Config.VAL_FROM_TRAIN,
            random_state=42,
            stratify=train_val_df['class']
        )
        
        splits = {
            'Training': train_df['class'].map({'sarc': 'Sarcastic', 'notsarc': 'Not Sarcastic'}).value_counts(),
            'Validation': val_df['class'].map({'sarc': 'Sarcastic', 'notsarc': 'Not Sarcastic'}).value_counts(),
            'Test': test_df['class'].map({'sarc': 'Sarcastic', 'notsarc': 'Not Sarcastic'}).value_counts()
        }
        
        fig = plt.figure(figsize=(12, 8))
        
        colors = ['lightblue', 'lightcoral']
        
        data = []
        for split_name, counts in splits.items():
            data.append({
                'Split': split_name,
                'Not Sarcastic': counts.get('Not Sarcastic', 0),
                'Sarcastic': counts.get('Sarcastic', 0)
            })
        
        df_plot = pd.DataFrame(data)
        ax = df_plot.plot(
            x='Split',
            y=['Not Sarcastic', 'Sarcastic'],
            kind='bar',
            color=colors,
            width=0.8
        )
        
        plt.title('Distribution of Classes Across Dataset Splits', pad=20, fontsize=12)
        plt.ylabel('Number of Samples', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, padding=3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('split_distribution_bars.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        pie_fig, pie_axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (split_name, counts) in enumerate(splits.items()):
            total = counts.sum()
            sizes = [counts['Not Sarcastic']/total*100, counts['Sarcastic']/total*100]
            pie_axes[idx].pie(sizes, labels=['Not Sarcastic', 'Sarcastic'],
                            autopct='%1.1f%%', colors=colors)
            pie_axes[idx].set_title(f'{split_name} Set')
        
        # Save  charts
        pie_fig.tight_layout(pad=3.0)
        pie_fig.savefig('split_distribution_pies.pdf', bbox_inches='tight', dpi=300)
        plt.close('all')
        
        # Print statistics
        print("\nDataset Distribution Analysis:")
        print("=" * 50)
        for split_name, counts in splits.items():
            print(f"\n{split_name} Set:")
            print("-" * 30)
            total = counts.sum()
            for class_name, count in counts.items():
                percentage = (count/total) * 100
                print(f"{class_name}: {count} samples ({percentage:.1f}%)")
        
        print("\nTotal Statistics:")
        print("-" * 30)
        print(f"Total samples: {total_samples}")
        print(f"Training samples: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
        print(f"Validation samples: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
        print(f"Test samples: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error in dataset visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    visualize_dataset_splits() 