
# Sarcasm Detection in Social Media

This project explores both classical and deep-learning methods for detecting sarcasm in social-media text, focusing on a hybrid **BERT-BiLSTM** architecture that combines pretrained contextual embeddings with sequential modeling. On the News Headlines dataset (26 709 examples from The Onion vs. mainstream sources), our best model achieves **93.0 % test accuracy** and a **0.93 macro-F1** score, outperforming prior hybrid CNN-LSTM baselines (89.7 % accuracy).

## Project Structure

```
├── data/
│   ├── raw/ … original JSON headlines  
│   └── processed/
│       └── preprocessed_news.csv  … 26,709 headlines, split 64%/16%/20%  
│           └── train: 18 316 | val: 4 579 | test: 5 724
├── models/
│   ├── bert_lstm_model.py     … BERT-BiLSTM implementation  
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
│   └── word_cloud.py          … word-cloud generation  
├── main.py                    … end-to-end training and evaluation script  
└── requirements.txt           … exact versions of PyTorch, Transformers, scikit-learn, spaCy, Optuna, etc.
```

## Dataset

We use the **News Headlines** corpus (Kaggle), containing 26 709 JSON‐formatted headlines labeled `sarc` or `notsarc`. After preprocessing and splitting, we have 18 316 training, 4 579 validation, and 5 724 test examples.

## Preprocessing

- **Text cleaning**: Unicode normalization, lowercase (for uncased BERT), removal/masking of URLs and user mentions, conversion of emojis to text, contraction expansion, stripping special characters.  
- **Classical pipeline**: Stop-word removal, lemmatization, TF-IDF vectorization (unigrams + bigrams, max_features=10 000).  
- **Transformer pipeline**: BERT tokenization (max_length=128), padding/truncation, attention masks.

## Models

### Classical Baselines

- **Logistic Regression** and **Linear SVM** on TF-IDF features  
- Hyperparameter tuning via 3-fold `GridSearchCV` optimizing macro-F1  
- Balanced class weights to mitigate skew  

### BERT-BiLSTM Hybrid

- **BERT-base-uncased** with first six layers frozen for efficiency  
- **Bidirectional LSTM** (hidden_size=256) on top of BERT’s last hidden states  
- **Intermediate linear layer** (256 units) + ReLU + dropout (0.269)  
- Final sigmoid output for binary classification  
- **Loss**: BCEWithLogitsLoss; **Optimizer**: AdamW (lr=4.20×10⁻⁵, weight_decay=0.0403)  
- **Early stopping** (patience=1) over 5 epochs; best model saved by lowest validation loss

## Hyperparameter Optimization

We ran **6 Optuna trials** (≈ 7 h) over this search space:

| Hyperparameter       | Search Space                      |
|:---------------------|:----------------------------------|
| learning_rate        | [1e−6, 1e−4] (log-uniform)        |
| batch_size           | {8, 16, 32}                       |
| lstm_hidden_size     | {256, 384, 512}                   |
| intermediate_size    | {128, 256}                        |
| dropout_rate         | [0.2, 0.5]                        |
| weight_decay         | [0.01, 0.05]                      |
| frozen_layers        | {6, 7, 8, 9}                      |

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

| Metric        | Not Sarc. | Sarc. | Overall |
|:--------------|:---------:|:-----:|:-------:|
| Precision     | 0.93      | 0.93  |         |
| Recall        | 0.94      | 0.92  |         |
| **F1-Score**  | 0.93      | 0.92  | **0.93**|
| **Accuracy**  |           |       | **0.93**|

### Comparison

| Model                              | Test Accuracy |
|:-----------------------------------|:--------------|
| Misra & Arora (CNN-LSTM hybrid)    | 89.7 %        |
| **This work (BERT-BiLSTM)**        | **93.0 %**    |

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

