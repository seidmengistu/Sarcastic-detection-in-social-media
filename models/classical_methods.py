import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils.config import Config
from utils.dataset_loader import load_data
from utils.analysis_data import plot_training_results, get_metrics_report
from utils.preprocessing import preprocess_text

def preprocess_and_save_data():
    """Load raw data, preprocess it, and save to processed directory"""
    try:
        print("Loading raw data...")
        raw_data_path = os.path.join(Config.DATA_DIR, "raw", "dataset_unificato.csv")
        df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(df)} rows")

        print("Preprocessing data...")
        # Convert labels
        df['class'] = df['class'].replace({"notsarc": 0, "sarc": 1})
        
        # Preprocess text with progress bar
        tqdm.pandas(desc="Processing texts")
        df['text'] = df['text'].progress_apply(lambda x: preprocess_text(str(x)))
        
        # Create processed directory if it doesn't exist
        os.makedirs(os.path.dirname(Config.PREPROCESSED_DATA_PATH), exist_ok=True)
        
        # Save preprocessed data
        print(f"Saving preprocessed data to {Config.PREPROCESSED_DATA_PATH}")
        df.to_csv(Config.PREPROCESSED_DATA_PATH, index=False)
        print("✓ Preprocessing complete")
        
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run():
    try:
        # Load data using utility function
        print("Loading data...")
        data = load_data(preprocessed=True)
        if data is None:
            return
            
        train_texts, train_labels = data['train']
        val_texts, val_labels = data['val']
        test_texts, test_labels = data['test']

        # Define models with configurations
        models_and_params = {
            "Logistic Regression": {
                'model': LogisticRegression(class_weight='balanced', max_iter=Config.NUM_EPOCHS * 1000),
                'params': {
                    'clf__C': [0.01, 0.1, 1, 10]
                }
            },
            "SVM": {
                'model': LinearSVC(class_weight='balanced'),
                'params': {
                    'clf__C': [0.1, 1, 10]
                }
            },
            "Random Forest": {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'clf__n_estimators': [100, 200],
                    'clf__max_depth': [None, 10, 20]
                }
            },
            "XGBoost": {
                'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'params': {
                    'clf__n_estimators': [100, 200],
                    'clf__max_depth': [3, 6]
                }
            }
        }

        # Train and evaluate each model
        for name, mp in models_and_params.items():
            print(f"\n🔍 Training {name}")

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', mp['model'])
            ])

            # Train and validate
            grid = GridSearchCV(
                pipeline,
                param_grid=mp['params'],
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            # Fit on training data
            grid.fit(train_texts, train_labels)

            print(f"\n✅ Best Parameters for {name}: {grid.best_params_}")

            # Evaluate on validation set
            val_preds = grid.predict(val_texts)
            print(f"\n📊 Validation Report for {name}:")
            print(get_metrics_report(val_labels, val_preds))

            # Evaluate on test set
            test_preds = grid.predict(test_texts)
            print(f"\n📈 Final Test Report for {name}:")
            print(get_metrics_report(test_labels, test_preds))

            # Save best model if needed
            # torch.save(grid.best_estimator_, f'best_{name.lower().replace(" ", "_")}.pkl')

    except Exception as e:
        print(f"Error in classical methods: {str(e)}")
        import traceback
        traceback.print_exc()

def test_model_pipeline():
    try:
        print("Testing Classical Models pipeline...")
        
        # Test preprocessing if needed
        if not os.path.exists(Config.PREPROCESSED_DATA_PATH):
            print("Generating preprocessed data...")
            df = preprocess_and_save_data()
            if df is None:
                return False
        
        # Test data loading
        data = load_data(preprocessed=True)
        train_texts, train_labels = data['train']
        print("✓ Data loading successful")
        print(f"Number of training examples: {len(train_texts)}")
        
        # Test one model with small sample
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        # Fit on small subset
        pipeline.fit(train_texts[:100], train_labels[:100])
        print("✓ Model training successful")
        
        # Test prediction
        pred = pipeline.predict(train_texts[:5])
        print("✓ Prediction successful")
        print(f"Sample predictions: {pred}")
        
        return True
        
    except Exception as e:
        print(f"Error in Classical Models test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_pipeline()
    else:
        run()
