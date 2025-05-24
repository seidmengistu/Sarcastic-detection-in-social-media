import os
import torch

class Config:
    # Essential paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Sarcasm_Headlines_Dataset_v2.json")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "preprocessed_news.csv")

    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # BERT Configuration
    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_HIDDEN_SIZE = 768
    BERT_MAX_LENGTH = 128
    BERT_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_bert_model.pt")
    
    # RoBERTa Configuration
    ROBERTA_MODEL_NAME = 'roberta-base'
    ROBERTA_HIDDEN_SIZE = 256
    ROBERTA_MAX_LENGTH = 128
    ROBERTA_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_roberta_model.pt")
    
    # Common training parameters
    NUM_EPOCHS = 3
    TRAIN_VAL_SIZE = 0.80
    VAL_FROM_TRAIN = 0.20
    
    # Model configuration
    HIDDEN_SIZE = 256
    INTERMEDIATE_SIZE = 128
    # Training hyperparameters
    LEARNING_RATE = 1.667347031092182e-05       
    DROPOUT_RATE = 0.4699231239759112
    WEIGHT_DECAY = 0.01
    
    # Training settings
    BATCH_SIZE = 8  # Reduced batch size
    ACCUMULATION_STEPS = 4  # Add gradient accumulation
    
    # Memory optimization
    PIN_MEMORY = True
    NUM_WORKERS = 0  # Reduce worker threads
    
    print(f"Using device: {DEVICE}")
    
    # RoBERTa configurations
    ROBERTA_LEARNING_RATE = 2e-5 
    ROBERTA_BATCH_SIZE = 8 
    ROBERTA_DROPOUT = 0.1
    ROBERTA_INTERMEDIATE_SIZE = 128  
