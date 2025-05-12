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
    """
    Load and split the dataset using a two-stage split approach:
    1. First split: 80% train+val, 20% test
    2. Second split: From the 80%, split into 80% train, 20% val
    """
    # Load the appropriate dataset
    data_path = Config.PREPROCESSED_DATA_PATH if preprocessed else Config.RAW_DATA_PATH
    df = pd.read_csv(data_path)
    
    # Extract features and labels
    texts = df['text'].values
    labels = df['class'].values
    
    # First split: separate test set (20% of total data)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, 
        labels,
        test_size=(1 - Config.TRAIN_VAL_SIZE),  # 20% for test
        random_state=42,
        stratify=labels
    )
    
    # Second split: split train_val into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=Config.VAL_FROM_TRAIN,  # 20% of train_val (16% of total)
        random_state=42,
        stratify=train_val_labels
    )
    
    # Print split sizes for verification
    print(f"\nDataset split sizes:")
    print(f"Training:   {len(train_texts)} ({len(train_texts)/len(texts):.1%})")
    print(f"Validation: {len(val_texts)} ({len(val_texts)/len(texts):.1%})")
    print(f"Test:       {len(test_texts)} ({len(test_texts)/len(texts):.1%})")
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }