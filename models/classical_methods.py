import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils.config import Config
from utils.dataset_loader import load_data
from utils.analysis_data import plot_training_results, get_metrics_report
from utils.preprocessing import preprocess_text

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

        # Convert string labels to integers and numpy arrays
        train_labels = np.array([1 if label == 'sarc' else 0 for label in train_labels])
        val_labels = np.array([1 if label == 'sarc' else 0 for label in val_labels])
        test_labels = np.array([1 if label == 'sarc' else 0 for label in test_labels])
        train_texts = np.array(train_texts)

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
        }

        # Store results for plotting
        all_results = {}

        # Train and evaluate each model
        for name, mp in models_and_params.items():
            print(f"\n{'='*20} Training {name} {'='*20}")

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', mp['model'])
            ])

            # Lists to store metrics for plotting
            train_losses = []
            val_losses = []
            train_metrics = []
            val_metrics = []

            # Create KFold cross-validator
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            # Train and validate
            grid = GridSearchCV(
                pipeline,
                param_grid=mp['params'],
                cv=kf,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            # Fit on training data
            grid.fit(train_texts, train_labels)

            # Store metrics from CV results
            for train_idx, val_idx in kf.split(train_texts):
                # Get predictions for this fold
                train_fold_pred = grid.predict(train_texts[train_idx])
                val_fold_pred = grid.predict(train_texts[val_idx])
                
                # Get metrics reports
                train_report = classification_report(
                    train_labels[train_idx],
                    train_fold_pred,
                    target_names=['Not Sarcastic', 'Sarcastic'],
                    output_dict=True
                )
                val_report = classification_report(
                    train_labels[val_idx],
                    val_fold_pred,
                    target_names=['Not Sarcastic', 'Sarcastic'],
                    output_dict=True
                )
                
                # Add true/pred pairs for confusion matrix
                train_report['true'] = train_labels[train_idx].tolist()
                train_report['pred'] = train_fold_pred.tolist()
                val_report['true'] = train_labels[val_idx].tolist()
                val_report['pred'] = val_fold_pred.tolist()
                
                # Store metrics
                train_metrics.append(train_report)
                val_metrics.append(val_report)
                
                # Use negative log loss as a proxy for loss
                train_losses.append(1 - train_report['accuracy'])
                val_losses.append(1 - val_report['accuracy'])

            print(f"\n✅ Best Parameters for {name}: {grid.best_params_}")

            # Evaluate on validation set
            val_preds = grid.predict(val_texts)
            print(f"\n📊 Validation Results for {name}:")
            print(classification_report(
                val_labels,
                val_preds,
                target_names=['Not Sarcastic', 'Sarcastic'],
                digits=4
            ))

            # Evaluate on test set
            test_preds = grid.predict(test_texts)
            print(f"\n📈 Test Results for {name}:")
            print(classification_report(
                test_labels,
                test_preds,
                target_names=['Not Sarcastic', 'Sarcastic'],
                digits=4
            ))

            # Store results for plotting
            all_results[name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }

            print("=" * 60)

        # Plot results for each model
        for name, results in all_results.items():
            save_path = f'classical_{name.lower().replace(" ", "_")}.png'
            plot_training_results(
                results['train_losses'],
                results['val_losses'],
                results['train_metrics'],
                results['val_metrics'],
                save_path=save_path
            )
            print(f"✓ Results plot saved as {save_path}")

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
