import json
import pandas as pd
import re
import emoji
import unicodedata
from tqdm import tqdm
import os
from config import Config

def preprocess_text(text: str, lowercase: bool = True) -> str:
    """
    Preprocess a single text entry
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
    """
    text = unicodedata.normalize("NFKC", text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    
    text = emoji.demojize(text, delimiters=(" :", ": "))
    
    if lowercase:
        text = text.lower()
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_news_dataset():
    """
    Preprocess the news sarcasm dataset by:
    1. Loading JSON data
    2. Converting to DataFrame
    3. Cleaning text
    4. Saving processed data
    """
    try:
        
        print("Loading news dataset...")
        data = []
        with open(Config.RAW_DATA_PATH, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        print("\nPreprocessing headlines...")
        tqdm.pandas()
        df['text'] = df['headline'].progress_apply(
            lambda x: preprocess_text(x, lowercase='uncased' in Config.BERT_MODEL_NAME)
        )
        
        df['class'] = df['is_sarcastic'].map({1: 'sarc', 0: 'notsarc'})
        
        final_df = df[['text', 'class']]
        
        print("\nSaving processed dataset...")
        final_df.to_csv(Config.PREPROCESSED_DATA_PATH, index=False)
        
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
        print(f"Error preprocessing news dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Processing news sarcasm dataset...")
    preprocess_news_dataset() 