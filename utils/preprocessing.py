import spacy
import re
import string
import emoji
import os
from tqdm import tqdm

# Load spaCy's English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def remove_sarcasm_indicators(text):
    """Remove explicit sarcasm indicators"""
    sarcasm_indicators = [
        'sarcastic', 'sarcasm', 'sarcast', 
        'sarcasms', 'sarcasmintended', 'sarcasmalert',
        'sarcasmoff', 'sarcasmon', '/s', '#s'
    ]
    pattern = r'\b(?:' + '|'.join(sarcasm_indicators) + r')\b'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

def clean_social_media_elements(text):
    """Clean social media specific elements"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove RT (retweet) indicator
    text = re.sub(r'\bRT\b', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def normalize_elongated_words(text):
    """Normalize elongated words (e.g., 'soooo' -> 'so')"""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def preprocess_text(
    text,
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=False,  # Keep stopwords as they might be important for sarcasm
    remove_digits=True,
    lemmatize=True,
    preserve_emojis=True,
):
    try:
        # Convert to string if not already
        text = str(text)
        
        # Handle emojis
        if preserve_emojis:
            text = emoji.demojize(text)
        
        # Clean social media elements
        text = clean_social_media_elements(text)
        
        # Remove sarcasm indicators
        text = remove_sarcasm_indicators(text)
        
        # Normalize elongated words
        text = normalize_elongated_words(text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove digits
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if remove_punctuation:
            # Preserve emojis and common symbols that might indicate sarcasm
            text = re.sub(r'[^\w\s:;()!?]', '', text)
        
        # Process with spaCy
        doc = nlp(text)
        
        # Lemmatize & optionally remove stopwords
        processed = []
        for token in doc:
            if remove_stopwords and token.is_stop:
                continue
            word = token.lemma_ if lemmatize else token.text
            processed.append(word)
        
        # Join tokens back into text
        processed_text = ' '.join(processed)
        
        # Final cleanup of whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text if processed_text else text  # Return original if empty
        
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return text

def process_dataset(input_path, output_path):
    """Process the entire dataset"""
    import pandas as pd
    import os
    
    try:
        print(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        print("Preprocessing texts...")
        tqdm.pandas(desc="Processing")
        df['text'] = df['text'].progress_apply(preprocess_text)
        
        # Remove empty texts
        initial_size = len(df)
        df = df[df['text'].str.strip().str.len() > 0]
        removed = initial_size - len(df)
        if removed > 0:
            print(f"Removed {removed} empty texts")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed dataset
        df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to {output_path}")
        print(f"Final dataset size: {len(df)} rows")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from utils.config import Config
    
    # Process the dataset
    process_dataset(
        input_path=os.path.join(Config.DATA_DIR, "raw", "dataset_unificato.csv"),
        output_path=Config.PREPROCESSED_DATA_PATH
    )
    
