# Sarcasm Detection in Social Media

This project explores classical, deep-learning, and hybrid methods for detecting sarcasm in social-media text, focusing on three main architectures:

1. A hybrid **BERT-BiLSTM** that combines pretrained contextual embeddings with sequential modeling
2. A hybrid **CNN-BiLSTM-Attention** that combines n-gram features with sequential modeling and attention mechanism
3. Classical methods (**SVM** and **Logistic Regression**) with TF-IDF features

On the News Headlines dataset (26,709 examples from The Onion vs. mainstream sources), our best models achieve:

- BERT-BiLSTM: **93.0% test accuracy** and **0.93 macro-F1** score
- CNN-BiLSTM-Attention: **91.2% test accuracy** and **0.91 macro-F1** score
- Classical Methods: **87.5% test accuracy** and **0.87 macro-F1** score

## Project Structure

```
├── data/
│   ├── raw/ … original JSON headlines
│   ├── embeddings/
│   │   └── glove.6B.300d.txt  … GloVe embeddings for hybrid model
│   └── processed/
│       └── preprocessed_news.csv  … 26,709 headlines, split 80/10/10
│           └── train: 21,367 | val: 2,671 | test: 2,671
├── models/
│   ├── bert_lstm_model.py     … BERT-BiLSTM implementation
│   ├── Hybrid_Neural_Network.py … CNN-BiLSTM-Attention implementation
│   └── classical_methods.py   … TF-IDF + Logistic Regression / SVM
├── utils/
│   ├── config.py              … all hyperparameters, paths, and device settings
│   ├── dataset_loader.py      … stratified train/val/test split
│   ├── preprocessing.py       … text cleaning, emoji demojization, URL masking
│   ├── preprocess_news.py     … news-specific normalization
│   ├── model_tuner.py         … Optuna hyperparameter search
│   ├── evaluation_utils.py    … metrics, classification reports, confusion matrices
│   ├── analysis_data.py       … loss/F1 plotting (training curves)
│   ├── data_visualization.py  … class distribution and exploratory plots
│   └── word_cloud.py         … word-cloud generation
├── main.py                    … end-to-end training and evaluation script
└── requirements.txt           … exact versions of PyTorch, Transformers, scikit-learn, spaCy, Optuna, etc.
```

## Dataset

The dataset is split into for CNN-Bilstm+attention:

- Training: 80% (21,367 samples)
- Validation: 10% (2,671 samples)
- Test: 10% (2,671 samples)

For the Bert_Bilstm 80% for Training and Validation and 20% for testing
- Training:64% of the 80%
- Validation:16% of the 80%
- Testing:20%
For classicals we used 3-fold cross validation.

All splits maintain class distribution through stratification.

## Preprocessing

- **Text cleaning**: Unicode normalization, lowercase (for uncased BERT), removal/masking of URLs and user mentions, conversion of emojis to text, contraction expansion, stripping special characters.
- **Classical pipeline**: Stop-word removal, lemmatization, TF-IDF vectorization (unigrams + bigrams, max_features=10 000).
- **Transformer pipeline**: BERT tokenization (max_length=128), padding/truncation, attention masks.

## Models

### 1. BERT-BiLSTM

- Combines BERT's contextual embeddings with bidirectional LSTM
- Uses attention mechanism for focusing on relevant parts of text
- Achieves state-of-the-art performance on the dataset

### 2. CNN-BiLSTM-Attention (Hybrid Neural Network)

- Combines CNN for n-gram feature extraction with BiLSTM for sequential modeling
- Uses GloVe embeddings (300d) for word representations
- Implements Bahdanau-style attention mechanism
- Features:
  - Multiple convolutional filters for capturing different n-gram patterns
  - Bidirectional LSTM for capturing long-range dependencies
  - Attention mechanism for focusing on relevant parts of text
  - Dropout for regularization
  - Grid search for hyperparameter optimization

### 3. Classical Methods

- Implements both SVM and Logistic Regression
- Uses TF-IDF vectorization for feature extraction
- Includes regularization and parameter tuning

## Hyperparameter Optimization

We ran **6 Optuna trials** (≈ 7 h) over this search space:

| Hyperparameter    | Search Space               |
| :---------------- | :------------------------- |
| learning_rate     | [1e−6, 1e−4] (log-uniform) |
| batch_size        | {8, 16, 32}                |
| lstm_hidden_size  | {256, 384, 512}            |
| intermediate_size | {128, 256}                 |
| dropout_rate      | [0.2, 0.5]                 |
| weight_decay      | [0.01, 0.05]               |
| frozen_layers     | {6, 7, 8, 9}               |

The best configuration (val loss = 0.1798) was:

- **lr** = 4.20×10⁻⁵
- **batch_size** = 16
- **hidden_size** = 256
- **intermediate_size** = 256
- **dropout** = 0.269
- **weight_decay** = 0.0403
- **frozen_layers** = 6

## Performance

### Validation (Epoch 2)

- **Train loss** ↓ from 0.2887 → 0.1585
- **Val loss** ↓ to 0.1756 (best)
- **Macro-F1** → 0.9326
- Precision/Recall for both classes ≈ 0.93

### Test

| Metric       | Not Sarc. | Sarc. | Overall  |
| :----------- | :-------: | :---: | :------: |
| Precision    |   0.93    | 0.93  |          |
| Recall       |   0.94    | 0.92  |          |
| **F1-Score** |   0.93    | 0.92  | **0.93** |
| **Accuracy** |           |       | **0.93** |

### Comparison

| Model                           | Test Accuracy |
| :------------------------------ | :------------ |
| Misra & Arora (CNN-LSTM hybrid) | 89.7 %        |
| **This work (BERT-BiLSTM)**     | **93.0 %**    |

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Preprocess data**
   ```bash
   python utils/preprocess_news.py
   python utils/preprocessing.py
   ```
3. **Visualize / Analyze**
   ```bash
   python utils/word_cloud.py
   python utils/data_visualization.py
   ```
4. **Hyperparameter tuning**
   ```bash
   python utils/model_tuner.py
   ```
5. **Train & evaluate**
   ```bash
   python main.py
   python -m utils.test_evaluation
   ```
