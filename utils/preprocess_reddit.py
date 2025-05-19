import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from .config import Config
from .preprocessing import preprocess_text

def preprocess_reddit_dataset():
    """
    Preprocess the Reddit sarcasm dataset by:
    1. Loading only necessary columns
    2. Cleaning and formatting text
    3. Combining parent and child comments
    4. Saving processed data
    """
    try:
        raw_path = os.path.join(Config.DATA_DIR, "raw", "train-balanced-sarcasm.csv")
        processed_path = os.path.join(Config.DATA_DIR, "processed", "preprocessed_dataset.csv")
        
        print("Loading Reddit dataset...")
        # Read only necessary columns
        df = pd.read_csv(
            raw_path,
            usecols=['label', 'comment', 'parent_comment']
        )
        
        print(f"Loaded {len(df)} samples")
        
        # Clean missing values
        df['parent_comment'].fillna("", inplace=True)
        
        print("Preprocessing comments...")
        tqdm.pandas()
        
        # Preprocess both comments
        df['comment'] = df['comment'].progress_apply(preprocess_text)
        df['parent_comment'] = df['parent_comment'].progress_apply(preprocess_text)
        
        # Remove empty comments after preprocessing
        df = df[df['comment'].str.strip() != ""]
        
        # Create combined text field
        df['text'] = df.apply(
            lambda x: f"{x['parent_comment']} [SEP] {x['comment']}" if x['parent_comment'] else x['comment'],
            axis=1
        )
        
        # Convert labels to match current format
        df['class'] = df['label'].map({0: 'notsarc', 1: 'sarc'})
        
        # Select final columns
        final_df = df[['text', 'class']]
        
        # Save processed data
        print(f"Saving {len(final_df)} processed samples...")
        final_df.to_csv(processed_path, index=False)
        
        # Print statistics
        print("\nDataset Statistics:")
        print("-" * 50)
        print(f"Total samples: {len(final_df)}")
        print("\nClass distribution:")
        print(final_df['class'].value_counts(normalize=True).round(3) * 100)
        print("\nSample processed texts:")
        print("-" * 50)
        print(final_df['text'].head())
        
        return True
        
    except Exception as e:
        print(f"Error preprocessing Reddit dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Processing Reddit sarcasm dataset...")
    preprocess_reddit_dataset() 