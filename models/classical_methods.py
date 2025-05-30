import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from utils.config import Config
from utils.dataset_loader import load_data
from utils.analysis_data import plot_classical_results
from utils.preprocessing import preprocess_text

def get_feature_importance(grid):
    """Extract feature importance from the model"""
    if hasattr(grid.best_estimator_.named_steps['clf'], 'coef_'):
        # Get the fitted vectorizer from the pipeline
        vectorizer = grid.best_estimator_.named_steps['tfidf']
        feature_names = vectorizer.get_feature_names_out()
        coefficients = grid.best_estimator_.named_steps['clf'].coef_[0]
        # Get top features with their names
        feature_importance = np.abs(coefficients)
        return feature_importance
    return []

def get_cv_results_by_c(grid):
    """Extract CV results organized by C value"""
    c_values = []
    cv_scores = []
    
    # Get all results
    for params, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        c_value = params['clf__C']
        if c_value not in c_values:
            c_values.append(c_value)
            cv_scores.append(score)
            
    # Sort by C value
    c_scores = sorted(zip(c_values, cv_scores), key=lambda x: x[0])
    return [x[0] for x in c_scores], [x[1] for x in c_scores]

def collect_model_results(grid, model_name, y_true, y_pred):
    """Collect all relevant metrics for plotting"""
    # Get CV results organized by C value
    c_values, cv_scores = get_cv_results_by_c(grid)
    
    results = {
        'c_values': c_values,
        'cv_scores': cv_scores,
        'feature_importance': get_feature_importance(grid),
        'precision': classification_report(y_true, y_pred, output_dict=True)['macro avg']['precision'],
        'recall': classification_report(y_true, y_pred, output_dict=True)['macro avg']['recall'],
        'f1': classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score'],
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return results

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
                'model': LogisticRegression(
                    class_weight='balanced',
                    max_iter=2 * 1000,
                    random_state=42
                ),
                'params': {
                    'clf__C': [0.001, 0.01, 0.1, 1, 10],
                    'clf__penalty': ['l1', 'l2'],
                    'clf__solver': ['liblinear', 'saga']
                }
            },
            "SVM": {
                'model': LinearSVC(
                    class_weight='balanced',
                    random_state=42
                ),
                'params': {
                    'clf__C': [0.001, 0.01, 0.1, 1, 10],
                    'clf__penalty': ['l2'], 
                    'clf__loss': ['hinge', 'squared_hinge']
                }
            },
        }

        # Store results for plotting
        all_results = {}

        # Train and evaluate each model
        for name, mp in models_and_params.items():
            print(f"\n{'='*20} Training {name} {'='*20}")

            # Create pipeline with TF-IDF and classifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                    norm='l2'
                )),
                ('clf', mp['model'])
            ])

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
            
            # Get predictions
            val_preds = grid.predict(val_texts)
            test_preds = grid.predict(test_texts)
            
            # Collect results for plotting
            all_results[name] = collect_model_results(
                grid, 
                name,
                test_labels,
                test_preds
            )

            print(f"\n✅ Best Parameters for {name}: {grid.best_params_}")
            print(f"\n📊 Test Results for {name}:")
            print(classification_report(
                test_labels,
                test_preds,
                target_names=['Not Sarcastic', 'Sarcastic'],
                digits=4
            ))

        # Plot results using new plotting function
        plot_classical_results(all_results, 'classical_models_comparison.png')
        print("✓ Results plot saved as classical_models_comparison.png")

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
