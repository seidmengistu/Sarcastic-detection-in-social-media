import spacy
import re
import emoji
from tqdm import tqdm
import pandas as pd
from .config import Config

# Load spaCy's English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Comprehensive text preprocessing."""
    try:
        text = str(text)
        text = emoji.demojize(text)
        
        # Clean social media elements
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\bRT\b', '', text)
        
        # Remove sarcasm indicators
        sarcasm_indicators = ['sarcastic', 'sarcasm', 'sarcast', '/s', '#s']
        pattern = r'\b(?:' + '|'.join(sarcasm_indicators) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # normalize elongated words
        text = re.sub(r'\d+', '', text)  # remove digits
        text = re.sub(r'[^\w\s:;()!?]', '', text)  # remove special chars
        
        # Process with spaCy
        doc = nlp(text)
        processed = [token.lemma_ for token in doc]
        
        return ' '.join(processed).strip()
        
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return text

def process_dataset():
    """Process and save dataset."""
    try:
        print(f"Loading dataset from {Config.RAW_DATA_PATH}")
        df = pd.read_csv(Config.RAW_DATA_PATH)
        print("we are here",df.head())
        print("Preprocessing texts...")
        tqdm.pandas(desc="Processing")
        df['text'] = df['text'].progress_apply(preprocess_text)
        
        df = df[df['text'].str.strip().str.len() > 0]
        df.to_csv(Config.PREPROCESSED_DATA_PATH, index=False)
        print(f"Processed dataset saved ({len(df)} rows)")
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

if __name__ == "__main__":
    process_dataset()
    
