import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import Config
from .preprocessing import preprocess_text

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
    """Load data and return train/val/test splits"""
    try:
        # Check if preprocessed data exists
        if not os.path.exists(Config.PREPROCESSED_DATA_PATH):
            print("Preprocessed data not found. Starting preprocessing...")
            df = preprocess_and_save_data()
            if df is None:
                raise Exception("Preprocessing failed")
        else:
            print(f"Loading data from: {Config.PREPROCESSED_DATA_PATH}")
            df = pd.read_csv(Config.PREPROCESSED_DATA_PATH)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df['text'].values, df['class'].values,
            test_size=(Config.VAL_SIZE + Config.TEST_SIZE),
            random_state=42,
            stratify=df['class']
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
    
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None