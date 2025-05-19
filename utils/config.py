import os
import torch

class Config:
    # Essential paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset_unificato.csv")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "preprocessed_dataset.csv")
    
    # Create necessary directories
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # BERT Configuration
    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_HIDDEN_SIZE = 768
    BERT_MAX_LENGTH = 256
    BERT_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_bert_model.pt")
    
    # RoBERTa Configuration
    ROBERTA_MODEL_NAME = 'roberta-base'
    ROBERTA_HIDDEN_SIZE = 768
    ROBERTA_MAX_LENGTH = 128
    ROBERTA_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_roberta_model.pt")
    
    # Common training parameters
    NUM_EPOCHS = 5
    TRAIN_VAL_SIZE = 0.80
    VAL_FROM_TRAIN = 0.20
    
    # Model configuration
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 128
    # Training hyperparameters
    LEARNING_RATE = 0.0005656087011158776        
    DROPOUT_RATE = 0.28220658563239703
    WEIGHT_DECAY = 0.01
    
    # Training settings
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8  
    print(f"Using device: {DEVICE}")
    
    # RoBERTa configurations
    ROBERTA_LEARNING_RATE = 1e-5 
    ROBERTA_BATCH_SIZE = 8 
    ROBERTA_DROPOUT = 0.3
    ROBERTA_INTERMEDIATE_SIZE = 128  