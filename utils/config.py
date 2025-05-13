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
    
    # Modified Training hyperparameters
    BASE_LEARNING_RATE = 2e-5  
    NUM_EPOCHS = 8  
    DROPOUT_RATE = 0.5  
    WEIGHT_DECAY = 0.01  
    GRADIENT_ACCUMULATION_STEPS = 2  
    EARLY_STOPPING_PATIENCE = 3  
    
    # Modified Training splits
    TRAIN_VAL_SIZE = 0.85  
    VAL_FROM_TRAIN = 0.15  
    
    # Modified batch sizes
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8 
    # Class weights to handle imbalance
    CLASS_WEIGHTS = torch.tensor([1.2, 0.8])  
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
    LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) / 32
    
    @staticmethod
    def get_batch_size():
        """Simple batch size selection based on device"""
        if torch.cuda.is_available():
            return 16 if 'T4' in torch.cuda.get_device_name(0) else 12
        return 8
    
    @staticmethod
    def cleanup_gpu_memory():
        """Basic GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
