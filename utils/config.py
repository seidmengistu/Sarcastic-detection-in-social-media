import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Data files
    RAW_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "dataset_unificato.csv")
    PREPROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed_dataset.csv")
    
    # Model configs
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Training splits
    TRAIN_SIZE = 0.6
    VAL_SIZE = 0.2
    TEST_SIZE = 0.2
