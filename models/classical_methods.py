import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm  # for progress bar
from utils.preprocessing import preprocess_text

def run():
    try:

        # ================ LOAD & PREPROCESS DATA ================
        PROJECT_ROOT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../."))
        DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
        
        # Define paths for input and preprocessed data
        INPUT_DATA_PATH = os.path.join(DATA_DIR, "processed", "dataset_unificato.csv")
        PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "preprocessed_dataset.csv")

        # Check if preprocessed data already exists
        if os.path.exists(PREPROCESSED_DATA_PATH):
            print("Loading preprocessed data...")
            df = pd.read_csv(PREPROCESSED_DATA_PATH)
            print(f"Loaded {len(df)} preprocessed rows")
        else:
            print("Loading raw data...")
            df = pd.read_csv(INPUT_DATA_PATH)
            total_rows = len(df)
            print(f"Total rows to process: {total_rows}")

            print("Preprocessing data...")
            df["text"] = df["text"].dropna()
            df['class'].replace({"notsarc": 0, "sarc": 1}, inplace=True)
            
            # Show progress during preprocessing
            tqdm.pandas(desc="Processing texts")
            df["processed_text"] = df["text"].progress_apply(lambda x: preprocess_text(x))

            # Save preprocessed data
            print("Saving preprocessed data...")
            df.to_csv(PREPROCESSED_DATA_PATH, index=False)
            print(f"Preprocessed data saved to: {PREPROCESSED_DATA_PATH}")

        # ================ SPLIT INTO train / val / test ===================
        print("Splitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            df["processed_text"], df["class"],
            test_size=0.2, stratify=df["class"], random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25, stratify=y_temp, random_state=42
        )

        # ================ MODELS & PARAM GRIDS ====================
        models_and_params = {
            "Logistic Regression": {
                'model': LogisticRegression(class_weight='balanced', max_iter=1000),
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

        # ================ TRAIN, TUNE & EVALUATE ====================
        for name, mp in models_and_params.items():
            print(f"\n🔍 Tuning {name}")

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', mp['model'])
            ])

            grid = GridSearchCV(
                pipeline,
                param_grid=mp['params'],
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            print(f"\n✅ Best Parameters for {name}: {grid.best_params_}")

            # ============ VALIDATION EVALUATION ============
            val_preds = grid.predict(X_val)
            print(f"\n📊 Validation Report for {name}:")
            print(classification_report(y_val, val_preds,
                  target_names=["Not Sarcastic", "Sarcastic"]))

            # ============ TEST EVALUATION ============
            test_preds = grid.predict(X_test)
            print(f"\n📈 Final Test Report for {name}:")
            print(classification_report(y_test, test_preds,
                  target_names=["Not Sarcastic", "Sarcastic"]))

    except Exception as e:
        print(f"Error in classical methods: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
