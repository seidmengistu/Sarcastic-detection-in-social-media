import os
import torch

class Config:
    # Essential paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset_unificato.csv")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "preprocessed_dataset.csv")
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    
    # Create necessary directories
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    
    # Model configuration
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128  
    HIDDEN_SIZE = 128 
    INTERMEDIATE_SIZE = 256  
    
    
    # Training hyperparameters
    LEARNING_RATE = 5e-5  
    NUM_EPOCHS = 5
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 0.02
    
    # Data splits
    TRAIN_VAL_SIZE = 0.85
    VAL_FROM_TRAIN = 0.15
    
    # Training settings
    BATCH_SIZE = 24 if torch.cuda.is_available() else 8
