import os
import torch

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    
    # Ensure directories exist
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Data files
    RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "dataset_unificato.csv")
    PREPROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed_dataset.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    
    # Model configs
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    
    # Training hyperparameters
    BASE_LEARNING_RATE = 5e-5
    NUM_EPOCHS = 5
    DROPOUT_RATE = 0.7
    WEIGHT_DECAY = 0.15
    GRADIENT_ACCUMULATION_STEPS = 4
    EARLY_STOPPING_PATIENCE = 2
    
    # Training splits
    TRAIN_VAL_SIZE = 0.8
    VAL_FROM_TRAIN = 0.2
    
    # GPU settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    PIN_MEMORY = torch.cuda.is_available()
    
    # Label mapping
    LABEL_MAP = {'notsarc': 0, 'sarc': 1}
    
    # Visualization settings
    FIGURE_SIZE = (15, 10)
    DPI = 100
    
    # Dynamic batch size and learning rate calculation
    BATCH_SIZE = 24 if torch.cuda.is_available() else 8
    LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) / 32
    
    @staticmethod
    def get_batch_size():
        """Simple batch size selection based on device"""
        if torch.cuda.is_available():
            return 24 if 'T4' in torch.cuda.get_device_name(0) else 16
        return 8
    
    @staticmethod
    def cleanup_gpu_memory():
        """Basic GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
