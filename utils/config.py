import os
import torch

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Essential paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Sarcasm_Headlines_Dataset_v2.json")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "preprocessed_news.csv")

    # BERT + LSTM sizes
    BERT_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_bert_model.pt")
    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_MAX_LENGTH = 128    
    LSTM_HIDDEN_SIZE = 384    
    INTERMEDIATE_SIZE = 192  
    TRAIN_VAL_SIZE = 0.80
    VAL_FROM_TRAIN = 0.20
    
    # Training hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 8e-6
    DROPOUT_RATE = 0.45
    WEIGHT_DECAY = 0.025

    # Plotting parameters
    FIGURE_SIZE = (10, 5)
    DPI = 100
