import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import Config
from .preprocessing import preprocess_text, process_dataset

def preprocess_and_save_data():
    """Load raw data, preprocess it, and save to processed directory"""
    try:
        print("Loading raw data...")
        raw_data_path = os.path.join(Config.DATA_DIR, "raw", "dataset_unificato.csv")
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} rows")

        print("Preprocessing data...")
        # Convert labels
        df['class'] = df['class'].replace({"notsarc": 0, "sarc": 1})
        
        # Preprocess text with progress bar
        tqdm.pandas(desc="Processing texts")
        df['text'] = df['text'].progress_apply(lambda x: preprocess_text(str(x)))
        
        # Create processed directory if it doesn't exist
        os.makedirs(os.path.dirname(Config.PREPROCESSED_DATA_PATH), exist_ok=True)
        
        # Save preprocessed data
        print(f"Saving preprocessed data to {Config.PREPROCESSED_DATA_PATH}")
        df.to_csv(Config.PREPROCESSED_DATA_PATH, index=False)
        print("✓ Preprocessing complete")
        
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_data(preprocessed=True):
    """Load and split dataset."""
    try:
        # Load appropriate dataset
        data_path = Config.PREPROCESSED_DATA_PATH if preprocessed else Config.RAW_DATA_PATH
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, 
            train_size=Config.TRAIN_VAL_SIZE, 
            random_state=42,
            stratify=df['class']
        )
        
        # Second split: train and validation
        train_df, val_df = train_test_split(
            train_val_df, 
            train_size=1-Config.VAL_FROM_TRAIN, 
            random_state=42,
            stratify=train_val_df['class']
        )
        
        # Print split sizes
        print("\nDataset split sizes:")
        print(f"Training:   {len(train_df)} ({len(train_df)/len(df):.1%})")
        print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
        print(f"Test:       {len(test_df)} ({len(test_df)/len(df):.1%})")
        
        return {
            'train': (train_df['text'].values, train_df['class'].values),
            'val': (val_df['text'].values, val_df['class'].values),
            'test': (test_df['text'].values, test_df['class'].values)
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None