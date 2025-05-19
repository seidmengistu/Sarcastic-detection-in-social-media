# Sarcasm Detection in Social Media

This project implements deep learning models and classical methods to detect sarcasm in social media text using BERT and RoBERTa with LSTM architectures.

## Project Structure

```
├── data/
│   ├── processed/
│   │   └── preprocessed_dataset.csv
│   └── raw/
│       └── dataset_unificato.csv
├── models/
│   ├── bert_lstm_model.py
│   └── roberta_lstm_model.py
└── └── classical_methods.py
   
     
├── utils/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── evaluation_utils.py
│   ├── preprocess_reddit.py
│   ├── test_evaluation.py
│   ├── model_tuner.py
│   ├── preprocessing.py
│   ├── evaluation_utils.py
│   └── word_cloud.py
│   └── analysis_data.py
├── main.py
└── requirements.txt
```

## Models

### 1. BERT-LSTM

- Uses BERT-base-uncased as the encoder
- Followed by a bidirectional LSTM layer
- Includes dropout and intermediate layers for better generalization

### 2. RoBERTa-BiLSTM

- Uses RoBERTa-base as the encoder
- Implements a bidirectional LSTM for sequence processing
- Features mixed precision training for efficiency
- Includes gradient clipping and memory optimization

## Features

- Mixed precision training (FP16)
- Gradient clipping for stability
- Memory optimization for GPU usage
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Word cloud generation for data analysis

## Usage

1. **Data Preprocessing**:

```bash
python utils/preprocess_reddit.py
```

2.**word cloud**

```bash
python utils/word_cloud.py
```

3.**model_tuner**

```bash
python utils/model_tuner.py
```

4. **Training Models**:

```bash
python main.py
```

5. **Evaluating Models**:if you want to evaluate the models with new data/with the available test set, you can use the following command:

```bash
python -m utils.test_evaluation --model bert    # Evaluate BERT model
python -m utils.test_evaluation --model roberta # Evaluate RoBERTa model
```





## Configuration

Key configurations in `utils/config.py`:

- Model architectures (BERT/RoBERTa)
- Training parameters
- Data paths
- GPU/CPU settings

## Requirements

- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib
- seaborn

## Model Performance

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Results are saved and visualized for both training and validation phases.

```
