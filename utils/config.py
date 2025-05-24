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
    
    # BERT Configuration
    BERT_BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_bert_model.pt")
    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_HIDDEN_SIZE = 768
    INTERMEDIATE_SIZE = 128
    BERT_MAX_LENGTH = 128
    TRAIN_VAL_SIZE = 0.80
    VAL_FROM_TRAIN = 0.20
    NUM_EPOCHS = 3
        
    # Training hyperparameters
    LEARNING_RATE = 1.667347031092182e-05       
    DROPOUT_RATE = 0.4699231239759112
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 8 

    # Plotting configuration
    FIGURE_SIZE = (15, 10)
    DPI = 100 