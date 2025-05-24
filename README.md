# Sarcasm Detection in Social Media

This project implements deep learning and classical machine learning methods to detect sarcasm in social media text using BERT-BiLSTM and traditional approaches.

## Project Structure

```
├── data/
│   ├── processed/
│   │   └── preprocessed_dataset.csv
│   └── raw/
│       └── dataset_unificato.csv
├── models/
│   ├── bert_lstm_model.py      # BERT-BiLSTM implementation
│   └── classical_methods.py
├── utils/
│   ├── config.py              # Configuration settings
│   ├── dataset_loader.py      # Data loading and splitting
│   ├── evaluation_utils.py    # Evaluation metrics and visualization
│   ├── preprocess_news.py     # News dataset preprocessing
│   ├── test_evaluation.py     # Model testing utilities
│   ├── model_tuner.py        # Hyperparameter optimization
│   ├── preprocessing.py       # Text preprocessing utilities
│   ├── analysis_data.py      # Training analysis and plotting
│   ├── data_visualization.py # Dataset visualization
│   └── word_cloud.py         # Word cloud generation
├── main.py
└── requirements.txt
```

## Models

### BERT-BiLSTM Architecture

- Uses BERT-base-uncased as the encoder for contextual embeddings
- Enhanced with a Bidirectional LSTM (BiLSTM) layer for sequence processing
  - Processes sequences in both forward and backward directions
  - Captures contextual dependencies from both past and future tokens
- Includes intermediate linear layer for feature transformation
- Employs dropout for regularization and overfitting prevention
- Features mixed precision training for computational efficiency
- Uses gradient checkpointing to optimize memory usage

### Classical Methods

- Logistic Regression with TF-IDF features
- Support Vector Machine (SVM) with TF-IDF features
- Includes grid search for hyperparameter optimization
- Balanced class weights for handling imbalanced data

## Features

- Comprehensive evaluation metrics
- Confusion matrix visualization
- Word cloud generation for data analysis
- Training progress visualization
- Dataset distribution analysis
- Hyperparameter optimization

## Usage

1. **Data Preprocessing**:

   ```bash
   # Preprocess the news headlines dataset
   python utils/preprocess_news.py

   # Run text preprocessing
   python utils/preprocessing.py
   ```

2. **Data Analysis and Visualization**:

   ```bash
   # Generate word clouds for analysis
   python utils/word_cloud.py

   # Visualize dataset distribution
   python utils/data_visualization.py
   ```

3. **Model Tuning**:

   ```bash
   # Run hyperparameter optimization for BERT-BiLSTM
   python utils/model_tuner.py
   ```

4. **Training Models**:

   ```bash
   # Train both BERT-BiLSTM and classical models
   python main.py
   ```

5. **Model Evaluation**:
   ```bash
   # Evaluate BERT-BiLSTM model
   python -m utils.test_evaluation
   ```

## Utility Files Description

- **config.py**: Central configuration file containing model parameters, paths, and training settings
- **dataset_loader.py**: Handles data loading, splitting into train/val/test sets, and preprocessing
- **evaluation_utils.py**: Contains functions for model evaluation, metrics calculation, and visualization
- **preprocess_news.py**: Specific preprocessing for news headlines dataset
- **test_evaluation.py**: Comprehensive model evaluation on test set
- **model_tuner.py**: Implements Optuna-based hyperparameter optimization
- **preprocessing.py**: Text preprocessing utilities including cleaning and normalization
- **analysis_data.py**: Training progress visualization and analysis tools
- **data_visualization.py**: Dataset statistics and distribution visualization
- **word_cloud.py**: Word cloud generation for text analysis

## Model Performance Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Results are saved and visualized for both training and validation phases.

## Configuration

Key configurations in `utils/config.py`:

- Model architecture parameters (BERT-BiLSTM settings)
- Training hyperparameters
- Data paths
- GPU/CPU settings
- Visualization settings

## Requirements

- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib
- seaborn
- wordcloud
- optuna
- spacy
